import sys
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))  # root directory

from portfolio_management.trade import PositionSide, Trade, TradeAction, PositionAdjustment


class Portfolio:
    def __init__(self, initial_cash: float, target_gmv: Optional[float] = None):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.target_gmv = target_gmv
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_history: Dict[str, List[PositionAdjustment]] = {}
        
        self.cash_history = []
        self.position_history = []
        self.nav_history = []

    def get_current_gmv(self, current_prices: Dict[str, float]) -> float:
        gmv = 0
        for ticker, trade in self.open_trades.items():
            price = current_prices.get(ticker, trade.entry_price)
            gmv += abs(trade.shares * price)
        return gmv

    def get_position_gmv(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get current GMV for each open position"""
        return {
            ticker: abs(trade.shares * current_prices[ticker])
            for ticker, trade in self.open_trades.items()
            if ticker in current_prices
        }

    def get_nmv(self, current_prices: Dict[str, float]) -> float:
        nmv = 0
        for ticker, trade in self.open_trades.items():
            price = current_prices.get(ticker, trade.entry_price)
            nmv += trade.shares * price * trade.side.value
        return nmv

    def _open_position(self, ticker: str, side: PositionSide, shares: float, price: float, date: datetime):
        assert self.cash - shares * price * side.value >= 0, 'You cannot open a position that results in negative cash'
        
        trade = Trade(
            ticker=ticker,
            side=side,
            entry_date=date,
            entry_price=price,
            shares=shares,
            action=TradeAction.OPEN
        )
        self.open_trades[ticker] = trade

        # Track adjustment
        if ticker not in self.trade_history:
            self.trade_history[ticker] = []

        self.trade_history[ticker].append(PositionAdjustment(
            date=date,
            action=TradeAction.OPEN,
            shares_delta=shares * side.value,
            price=price,
            resulting_shares=shares
        ))

        # Update cash (for long we pay, for short we receive)
        self.cash -= shares * price * side.value

    def _add_to_position(self, ticker: str, target_shares: float, price: float, date: datetime):
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        shares_delta = target_shares - trade.shares

        # Calculate new weighted average entry price
        old_value = trade.shares * trade.entry_price
        new_value = shares_delta * price
        trade.entry_price = (old_value + new_value) / target_shares
        trade.shares = target_shares

        # Track adjustment
        self.trade_history[ticker].append(PositionAdjustment(
            date=date,
            action=TradeAction.OPEN,
            shares_delta=shares_delta * trade.side.value,
            price=price,
            resulting_shares=target_shares
        ))

        # Update cash
        self.cash -= shares_delta * price * trade.side.value

    def _reduce_position(self, ticker: str, target_shares: float,
                         price: float, date: datetime):
        """Reduce existing position (partial close/cover)"""
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        shares_delta = target_shares - trade.shares  # Negative value
        shares_closed = abs(shares_delta)

        assert self.cash - shares_delta * price * trade.side.value >= 0, 'You cannot open a position that results in negative cash'

        # Create a closed trade record for the partial exit
        partial_trade = Trade(
            ticker=ticker,
            side=trade.side,
            entry_date=trade.entry_date,
            entry_price=trade.entry_price,
            shares=shares_closed,
            exit_date=date,
            exit_price=price,
            action=TradeAction.CLOSE
        )
        self.closed_trades.append(partial_trade)

        # Update the open position
        trade.shares = target_shares

        # Track adjustment - ADD THIS
        self.trade_history[ticker].append(PositionAdjustment(
            date=date,
            action=TradeAction.CLOSE,
            shares_delta=shares_delta * trade.side.value,  # Negative value
            price=price,
            resulting_shares=target_shares
        ))

        # Update cash (we get cash back)
        self.cash -= shares_delta * price * trade.side.value

    def _close_position(self, ticker: str, price: float, date: datetime,
                        action: TradeAction = TradeAction.CLOSE):
        """Close existing position completely"""
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        trade.exit_date = date
        trade.exit_price = price
        trade.action = action

        # Track adjustment
        self.trade_history[ticker].append(PositionAdjustment(
            date=date,
            action=action,
            shares_delta=-trade.shares * trade.side.value,
            price=price,
            resulting_shares=0
        ))

        # Settle cash
        self.cash += trade.shares * price * trade.side.value

        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[ticker]

    def update_positions(self, target_weights: Dict[str, float],
                    prices: Dict[str, float],
                    date: datetime):
        total_weight = sum(abs(w) for w in target_weights.values())
        
        if total_weight == 0:
            for ticker in list(self.open_trades.keys()):
                if ticker in prices:
                    self._close_position(ticker, prices[ticker], date, TradeAction.CLOSE)
            return
        
        all_tickers = set(list(self.open_trades.keys()) + list(target_weights.keys()))
        
        # PASS 1: Close/reduce positions first to free up cash
        for ticker in all_tickers:
            if ticker not in prices:
                continue
            
            target_weight = target_weights.get(ticker, 0.0)
            current_trade = self.open_trades.get(ticker)
            
            if not current_trade:
                continue
            
            target_notional = target_weight * self.target_gmv
            target_shares = round(abs(target_notional / prices[ticker])) if target_notional != 0 else 0
            target_side = PositionSide.LONG if target_weight > 0 else PositionSide.SHORT
            
            # Close positions with no target
            if abs(target_weight) < 1e-6:
                self._close_position(ticker, prices[ticker], date, TradeAction.CLOSE)
            
            # Close positions that need to flip sides
            elif current_trade.side != target_side:
                self._close_position(ticker, prices[ticker], date, TradeAction.CLOSE)
            
            # Reduce positions that are too large
            elif target_shares < current_trade.shares:
                self._reduce_position(ticker, target_shares, prices[ticker], date)
        
        # PASS 2: Open/add positions after cash is freed up
        for ticker in all_tickers:
            if ticker not in prices:
                continue
            
            target_weight = target_weights.get(ticker, 0.0)
            current_trade = self.open_trades.get(ticker)
            
            if abs(target_weight) < 1e-6:
                continue
            
            target_notional = target_weight * self.target_gmv
            target_shares = round(abs(target_notional / prices[ticker])) if target_notional != 0 else 0
            target_side = PositionSide.LONG if target_weight > 0 else PositionSide.SHORT
            
            # Open new position
            if not current_trade:
                self._open_position(ticker, target_side, target_shares, prices[ticker], date)
            
            # Add to existing position
            elif target_shares > current_trade.shares:
                self._add_to_position(ticker, target_shares, prices[ticker], date)

    def record_nav(self, date: datetime, current_prices: Dict[str, float]):
        current_nav = self.get_portfolio_value(current_prices)
        
        self.nav_history.append({
            'date' : date,
            'nav': current_nav})

    def record_cash(self, date: datetime):
        self.cash_history.append({
            'date': date,
            'cash': self.cash
        })

    def record_position(self, date: datetime, tickers: List[str], prices: Dict[str, float]):
        for ticker in tickers:
            shares = self.open_trades[ticker].shares * self.open_trades[ticker].side.value if ticker in self.open_trades else 0
            self.position_history.append({
                'date': date,
                'ticker': ticker,
                'shares': shares,
                'price': prices.get(ticker, 0)
            })

    def get_latest_cash(self):
        return self.cash
    
    def get_latest_positions(self, tickers: List[str]) -> Dict[str, float]:
        return {
            ticker: self.open_trades[ticker].shares * self.open_trades[ticker].side.value
            if ticker in self.open_trades else 0
            for ticker in tickers
        }

    def get_open_pnl(self, current_prices: Dict[str, float]) -> float:
        pnl = 0
        for ticker, trade in self.open_trades.items():
            if ticker in current_prices:
                price_change = current_prices[ticker] - trade.entry_price
                pnl += trade.shares * price_change * trade.side.value
        return pnl

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Total portfolio value (NAV)"""
        # Calculate market value of all positions
        position_value = 0
        for ticker, trade in self.open_trades.items():
            if ticker in current_prices:
                current_market_value = trade.shares * \
                    current_prices[ticker] * trade.side.value
                position_value += current_market_value

        nav = self.cash + position_value
        return nav
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        total_trades = 0

        volumes = []
        tot_volume = 0

        for ticker in self.trade_history.keys():
            total_trades += len(self.trade_history[ticker])
            
            ticker_volume = 0
            for trade in self.trade_history[ticker]:
                ticker_volume += abs(trade.shares_delta * trade.price)
            volumes.append({'ticker' : ticker, 'volume' : ticker_volume})
            tot_volume += ticker_volume

        pnl = self.get_open_pnl(current_prices)

        for trade in self.closed_trades:
            pnl += trade.pnl
        
        return_per_trade = pnl/tot_volume

        daily_return = np.diff(self.nav_history)/self.nav_history[:-1]
        mean_return = np.mean(daily_return)
        std_return = np.std(daily_return)

        max_dd, max_dd_start, max_dd_end = self._calculate_max_drawdown(self.nav_history)

        return {
            'pnl' : pnl/self.initial_cash,
            'max_dd' : max_dd,
            'max_dd_start' : max_dd_start,
            'max_dd_end' : max_dd_end,
            'volume_trade' : tot_volume,
            'return_per_trade' : return_per_trade,
            'mean_return' : mean_return,
            'std_return' : std_return, 
            'sharpe' : mean_return/std_return
        }

    def _calculate_max_drawdown(self, nav: Dict[datetime, float]) -> Tuple:
        # Sort by date
        sorted_items = sorted(nav.items())
        dates = [item[0] for item in sorted_items]
        navs = [item[1] for item in sorted_items]

        peak_idx = 0
        max_dd = 0
        max_dd_start = dates[0]
        max_dd_end = dates[0]
        temp_peak_idx = 0

        for i in range(1, len(navs)):
            if navs[i] > navs[temp_peak_idx]:
                temp_peak_idx = i
            
            drawdown = (navs[i] - navs[temp_peak_idx]) / navs[temp_peak_idx]
            
            if drawdown < max_dd:
                max_dd = drawdown
                peak_idx = temp_peak_idx
                max_dd_start = dates[peak_idx]
                max_dd_end = dates[i]

        return max_dd, max_dd_start, max_dd_end
    
    def get_position_details(self, current_prices: Dict[str, float]) -> List[Dict]:
        positions = []
        for ticker, trade in self.open_trades.items():
            current_price = current_prices.get(ticker, trade.entry_price)
            unrealized_pnl = (current_price - trade.entry_price) * \
                trade.shares * trade.side.value
            notional = abs(trade.shares * current_price)

            positions.append({
                'ticker': ticker,
                'shares': trade.shares * trade.side.value,
                'cost_basis': trade.entry_price,
                'current_price': current_price,
                'notional': notional,
                'pnl': unrealized_pnl,
                'return': unrealized_pnl / (trade.shares * trade.entry_price) if trade.shares * trade.entry_price > 0 else 0,
            })
        return positions

    def get_position_history(self, ticker: str) -> List[PositionAdjustment]:
        return self.trade_history.get(ticker, [])

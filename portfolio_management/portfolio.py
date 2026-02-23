# autopep8: off
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.trade import (PositionAdjustment, PositionSide,
                                        Trade, TradeAction)

# autopep8: on


class Portfolio:
    def __init__(self, initial_cash: float,
                 target_gmv: Optional[float] = None):
        '''Initialise the portfolio with a starting cash balance.

        Parameters
        ----------
        initial_cash : float
            Starting cash balance.
        target_gmv : float, optional
            Target gross market value cap used for position sizing.
        '''
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
        '''Return the total gross market value of all open positions.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        float
            Sum of absolute notional values across all open positions.
        '''
        gmv = 0
        for ticker, trade in self.open_trades.items():
            price = current_prices.get(ticker, trade.entry_price)
            gmv += abs(trade.shares * price)
        return gmv

    def get_position_gmv(
            self, current_prices: Dict[str, float]) -> Dict[str, float]:
        '''Return the current gross market value for each open position.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        dict of str -> float
            Mapping of ticker -> absolute notional value for open positions
            present in current_prices.
        '''
        return {
            ticker: abs(trade.shares * current_prices[ticker])
            for ticker, trade in self.open_trades.items()
            if ticker in current_prices
        }

    def get_nmv(self, current_prices: Dict[str, float]) -> float:
        '''Return the net market value of all open positions.

        Long positions contribute positively, short positions negatively.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        float
            Signed net market value.
        '''
        nmv = 0
        for ticker, trade in self.open_trades.items():
            price = current_prices.get(ticker, trade.entry_price)
            nmv += trade.shares * price * trade.side.value
        return nmv

    def _open_position(self, ticker: str, side: PositionSide,
                       shares: float, price: float, date: datetime):
        '''Open a new position and deduct its cost from cash.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        side : PositionSide
            LONG or SHORT.
        shares : float
            Number of shares to buy/sell short.
        price : float
            Execution price per share.
        date : datetime
            Trade date.
        '''
        assert self.cash - shares * price * \
            side.value >= 0, 'You cannot open a position that results in negative cash'

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

    def _add_to_position(
            self, ticker: str, target_shares: float, price: float, date: datetime):
        '''Increase an existing position and update the weighted average entry price.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        target_shares : float
            Desired final share count (must exceed the current position size).
        price : float
            Execution price per share for the additional shares.
        date : datetime
            Trade date.
        '''
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        shares_delta = target_shares - trade.shares

        # Calculate new weighted average entry price
        existing_cost = trade.shares * trade.entry_price
        additional_cost = shares_delta * price
        trade.entry_price = (existing_cost + additional_cost) / target_shares
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
        '''Partially close an existing position down to the target share count.

        Creates a closed trade record for the exited portion and returns the
        proceeds to cash.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        target_shares : float
            Desired remaining share count (must be less than the current position size).
        price : float
            Execution price per share.
        date : datetime
            Trade date.
        '''
        if ticker not in self.open_trades:
            return

        trade = self.open_trades[ticker]
        shares_delta = target_shares - trade.shares  # Negative value
        shares_closed = abs(shares_delta)

        assert self.cash - shares_delta * price * \
            trade.side.value >= 0, 'You cannot reduce a position that results in negative cash'

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

        # Track adjustment
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
        '''Fully close an existing position and return the proceeds to cash.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        price : float
            Execution price per share.
        date : datetime
            Trade date.
        action : TradeAction, optional
            Action label to record on the trade (defaults to CLOSE).
        '''
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
        '''Reconcile current positions against target weights and execute required trades.

        Performs a two-pass update: first closes or reduces positions to free up cash,
        then opens or adds to positions.

        Parameters
        ----------
        target_weights : dict of str -> float
            Desired portfolio weights (positive = long, negative = short).
            Weights are scaled against the effective GMV (min of target_gmv and NAV).
        prices : dict of str -> float
            Execution prices for each ticker.
        date : datetime
            Execution date applied to all trades.
        '''
        total_weight = sum(abs(w) for w in target_weights.values())

        current_nav = self.get_portfolio_value(prices)
        effective_gmv = min(self.target_gmv, current_nav)

        if total_weight == 0:
            for ticker in list(self.open_trades.keys()):
                if ticker in prices:
                    self._close_position(
                        ticker, prices[ticker], date, TradeAction.CLOSE)
            return

        all_tickers = set(list(self.open_trades.keys()) +
                          list(target_weights.keys()))

        # PASS 1: Close/reduce positions first to free up cash
        for ticker in all_tickers:
            if ticker not in prices:
                continue

            target_weight = target_weights.get(ticker, 0.0)
            current_trade = self.open_trades.get(ticker)

            if not current_trade:
                continue

            target_notional = target_weight * effective_gmv
            target_shares = int(
                abs(target_notional / prices[ticker])) if target_notional != 0 else 0
            target_side = PositionSide.LONG if target_weight > 0 else PositionSide.SHORT

            # Close positions with no target
            if abs(target_weight) < 1e-6:
                self._close_position(
                    ticker, prices[ticker], date, TradeAction.CLOSE)

            # Close positions that need to flip sides
            elif current_trade.side != target_side:
                self._close_position(
                    ticker, prices[ticker], date, TradeAction.CLOSE)

            # Reduce positions that are too large
            elif target_shares < current_trade.shares:
                self._reduce_position(
                    ticker, target_shares, prices[ticker], date)

        # PASS 2: Open/add positions after cash is freed up
        for ticker in all_tickers:
            if ticker not in prices:
                continue

            target_weight = target_weights.get(ticker, 0.0)
            current_trade = self.open_trades.get(ticker)

            if abs(target_weight) < 1e-6:
                continue

            target_notional = target_weight * effective_gmv
            target_shares = int(
                abs(target_notional / prices[ticker])) if target_notional != 0 else 0
            target_side = PositionSide.LONG if target_weight > 0 else PositionSide.SHORT

            # Open new position
            if not current_trade:
                self._open_position(ticker, target_side,
                                    target_shares, prices[ticker], date)

            # Add to existing position
            elif target_shares > current_trade.shares:
                self._add_to_position(
                    ticker, target_shares, prices[ticker], date)

    def record_nav(self, date: datetime, current_prices: Dict[str, float]):
        '''Append the current NAV to nav_history.

        Parameters
        ----------
        date : datetime
            Snapshot date.
        current_prices : dict of str -> float
            Current price for each ticker used to value open positions.
        '''
        current_nav = self.get_portfolio_value(current_prices)

        self.nav_history.append({
            'date': date,
            'nav': current_nav})

    def record_cash(self, date: datetime):
        '''Append the current cash balance to cash_history.

        Parameters
        ----------
        date : datetime
            Snapshot date.
        '''
        self.cash_history.append({
            'date': date,
            'cash': self.cash
        })

    def record_position(self, date: datetime,
                        tickers: List[str], prices: Dict[str, float]):
        '''Append current share counts and prices for a list of tickers to position_history.

        Tickers not currently held are recorded with 0 shares.

        Parameters
        ----------
        date : datetime
            Snapshot date.
        tickers : list of str
            Tickers to record.
        prices : dict of str -> float
            Current price for each ticker.
        '''
        for ticker in tickers:
            shares = self.open_trades[ticker].shares * \
                self.open_trades[ticker].side.value if ticker in self.open_trades else 0
            self.position_history.append({
                'date': date,
                'ticker': ticker,
                'shares': shares,
                'price': prices.get(ticker, 0)
            })

    def get_latest_cash(self):
        '''Return the current cash balance.'''
        return self.cash

    def get_latest_positions(self, tickers: List[str]) -> Dict[str, float]:
        '''Return signed share counts for a list of tickers.

        Long positions are positive, short positions are negative, flat positions are 0.

        Parameters
        ----------
        tickers : list of str
            Tickers to query.

        Returns
        -------
        dict of str -> float
            Signed share count per ticker.
        '''
        return {
            ticker: self.open_trades[ticker].shares *
            self.open_trades[ticker].side.value
            if ticker in self.open_trades else 0
            for ticker in tickers
        }

    def get_open_pnl(self, current_prices: Dict[str, float]) -> float:
        '''Return the total unrealised PnL across all open positions.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        float
            Sum of unrealised PnL for all open trades.
        '''
        pnl = 0
        for ticker, trade in self.open_trades.items():
            if ticker in current_prices:
                price_change = current_prices[ticker] - trade.entry_price
                pnl += trade.shares * price_change * trade.side.value
        return pnl

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        '''Return the total portfolio value (NAV): cash plus marked-to-market position value.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        float
            Net asset value.
        '''
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
        '''Return a summary dict of key portfolio performance metrics.

        Includes PnL, NAV, returns, Sharpe ratio, max drawdown, and turnover statistics.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current prices used to mark open positions to market.

        Returns
        -------
        dict
            Keys: 'pnl', 'final_nav', 'return', 'annual_return', 'max_dd',
            'max_dd_start', 'max_dd_end', 'max_dd_duration', 'volume_trade',
            'return_per_trade', 'mean_return', 'std_return', 'sharpe'.
        '''
        tot_volume = 0

        for ticker in self.trade_history.keys():
            ticker_volume = 0
            for trade in self.trade_history[ticker]:
                ticker_volume += abs(trade.shares_delta * trade.price)
            tot_volume += ticker_volume

        pnl = self.get_open_pnl(current_prices)

        for trade in self.closed_trades:
            pnl += trade.pnl

        return_per_trade = pnl / tot_volume

        nav_history = [entry['nav'] for entry in self.nav_history]

        daily_return = np.diff(nav_history) / nav_history[:-1]
        mean_return = np.mean(daily_return)
        std_return = np.std(daily_return)

        # Calculate annual return
        total_trading_days = len(nav_history)
        annual_return = (1 + pnl / self.initial_cash) ** (252 /
                                                          total_trading_days) - 1 if total_trading_days > 0 else 0

        max_dd_dict = self._calculate_max_drawdown(
            self.nav_history)

        return {
            'pnl': pnl,
            'final_nav': self.nav_history[-1]['nav'],
            'return': pnl / self.initial_cash,
            'annual_return': annual_return,
            'max_dd': max_dd_dict['max_drawdown'],
            'max_dd_start': max_dd_dict['start_date'],
            'max_dd_end': max_dd_dict['end_date'],
            'max_dd_duration': max_dd_dict['duration_days'],
            'volume_trade': tot_volume,
            'return_per_trade': return_per_trade,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe': mean_return / std_return
        }

    def _calculate_max_drawdown(self, daily_nav: List[Dict]) -> Dict:
        '''Calculate the maximum drawdown from a NAV history list.

        Parameters
        ----------
        daily_nav : list of dict
            Each element must contain 'nav' (float) and 'date' keys.

        Returns
        -------
        dict
            Keys: 'max_drawdown' (float fraction), 'start_date', 'end_date',
            'duration_days' (int).
        '''
        nav_series = np.array([d['nav'] for d in daily_nav])
        dates = [d['date'] for d in daily_nav]

        cummax = np.maximum.accumulate(nav_series)
        drawdowns = (cummax - nav_series) / cummax

        max_dd = np.max(drawdowns)
        max_dd_idx = np.argmax(drawdowns)  # End of drawdown

        # Find start of drawdown (last peak before max_dd_idx)
        peak_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if nav_series[i] == cummax[i]:
                peak_idx = i
                break

        return {
            'max_drawdown': max_dd,
            'start_date': dates[peak_idx],
            'end_date': dates[max_dd_idx],
            'duration_days': (dates[max_dd_idx] - dates[peak_idx]).days
        }

    def get_position_details(
            self, current_prices: Dict[str, float]) -> List[Dict]:
        '''Return a list of dicts describing each open position.

        Each dict contains: 'ticker', 'shares' (signed), 'cost_basis',
        'current_price', 'notional', 'pnl' (unrealised), and 'return'.

        Parameters
        ----------
        current_prices : dict of str -> float
            Current price for each ticker.

        Returns
        -------
        list of dict
        '''
        positions = []
        for ticker, trade in self.open_trades.items():
            current_price = current_prices.get(ticker, trade.entry_price)
            unrealized_pnl = (current_price - trade.entry_price) * \
                trade.shares * trade.side.value
            notional = trade.side.value * trade.shares * current_price

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
        '''Return the list of PositionAdjustment records for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        list of PositionAdjustment
            All adjustments for the ticker in chronological order,
            or an empty list if no history exists.
        '''
        return self.trade_history.get(ticker, [])

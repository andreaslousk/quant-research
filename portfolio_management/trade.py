# autopep8: off
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

# autopep8: on


class PositionSide(Enum):
    LONG = 1
    SHORT = -1


class TradeAction(Enum):
    OPEN = 'open'
    CLOSE = 'close'


@dataclass
class Trade:
    ticker: str
    side: PositionSide
    entry_date: datetime
    entry_price: float
    shares: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    action: TradeAction = TradeAction.OPEN

    @property
    def is_open(self) -> bool:
        '''Return True if the trade has not yet been closed.'''
        return self.exit_date is None

    @property
    def notional_value(self) -> float:
        '''Return the absolute notional value of the trade at its entry (or exit) price.'''
        price = self.exit_price if not self.is_open else self.entry_price
        return abs(self.shares * price)

    @property
    def pnl(self) -> Optional[float]:
        '''Return the realised PnL for a closed trade, or None if the trade is still open.'''
        if not self.is_open:
            price_change = self.exit_price - self.entry_price
            # For shorts, side.value = -1 inverts the sign so a price decline yields positive PnL
            return self.shares * price_change * self.side.value
        return None

    @property
    def return_pct(self) -> Optional[float]:
        '''Return the percentage return for a closed trade, or None if still open.'''
        if not self.is_open:
                price_change = self.exit_price - self.entry_price
                return (price_change / self.entry_price) * self.side.value
        return None


@dataclass
class PositionAdjustment:
    date: datetime
    action: TradeAction
    shares_delta: float
    price: float
    resulting_shares: float

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class AccountInfo:
    initial_capital: float = 0.0
    dynamic_rights: float = 0.0
    total_profit: float = 0.0
    rate: float = 0.0
    fee: float = 0.0


@dataclass
class BacktestConfig:
    show_kline: bool = True


@dataclass
class TradeRecord:
    order_id: int = 0
    trade_time: Any = None
    instrument_id: str = ""
    direction: str = "0"
    offset: str = "0"
    volume: int = 0
    order_price: float = 0.0
    trade_price: float = 0.0
    fee: float = 0.0
    pnl: float = 0.0
    memo: str = ""


@dataclass
class InstrumentData:
    volume_multiple: float = 1.0


class BacktestingResult:
    def __init__(
        self,
        account: Optional[AccountInfo] = None,
        config: Optional[BacktestConfig] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        klines: Optional[List[Dict]] = None,
        trade_log: Optional[List[TradeRecord]] = None,
        df: Optional[pd.DataFrame] = None,
        instruments_data: Optional[Dict[str, InstrumentData]] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
        benchmark_symbol: str = "",
        trade_start_date: Optional[str] = None,
        instrument_close_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.account = account or AccountInfo()
        self.config = config or BacktestConfig()
        self.strategy_params = strategy_params or {}
        self.klines = klines or []
        self.trade_log = trade_log or []
        self.df = df
        self.instruments_data = instruments_data or {}
        self.benchmark_df = benchmark_df
        self.benchmark_symbol = benchmark_symbol
        self.trade_start_date: Optional[str] = trade_start_date
        self.instrument_close_prices = instrument_close_prices or {}
        self.turnover: float = 0.0
        self.total_volume: int = 0
        self.total_trading_days: int = 0
        self.pnl_days: tuple = (0, 0)
        self._data_prepared = False

    def prepare_data(self):
        if self._data_prepared:
            return
        if self.df is not None and not self.df.empty:
            self._compute_daily_metrics()
        self._data_prepared = True

    def _compute_daily_metrics(self):
        if self.df is None or self.df.empty:
            return
        if "PnL" in self.df.columns:
            pnl_series = self.df["PnL"]
            win_days = int((pnl_series > 0).sum())
            loss_days = int((pnl_series < 0).sum())
            self.pnl_days = (win_days, loss_days)
        self.total_trading_days = len(self.df)

    def annual_return(self, initial: float, final: float) -> float:
        if initial <= 0 or self.total_trading_days <= 0:
            return 0.0
        total_return = final / initial - 1
        years = self.total_trading_days / 252
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    def max_drawdown(self) -> float:
        if self.df is None or self.df.empty or "PortfolioValue" not in self.df.columns:
            return 0.0
        equity = self.df["PortfolioValue"].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(np.min(drawdown))

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if self.df is None or self.df.empty:
            return 0.0
        if "PnL" not in self.df.columns or "PortfolioValue" not in self.df.columns:
            return 0.0
        portfolio_values = self.df["PortfolioValue"].values
        pnl_values = self.df["PnL"].values
        if len(pnl_values) < 2 or len(portfolio_values) < 2:
            return 0.0
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            prev_value = portfolio_values[i - 1]
            if prev_value > 0:
                daily_returns.append(pnl_values[i] / prev_value)
            else:
                daily_returns.append(0.0)
        if len(daily_returns) < 2:
            return 0.0
        returns_arr = np.array(daily_returns)
        excess_returns = returns_arr - risk_free_rate / 252
        mean_excess = np.mean(excess_returns)
        std_ret = np.std(excess_returns, ddof=1)
        if std_ret == 0:
            return 0.0
        return float(mean_excess / std_ret * np.sqrt(252))

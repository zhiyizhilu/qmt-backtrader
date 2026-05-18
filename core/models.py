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


@dataclass
class BacktestingResult:
    account: AccountInfo = None
    config: BacktestConfig = None
    strategy_params: Dict[str, Any] = None
    klines: List[Dict] = None
    trade_log: List[TradeRecord] = None
    df: Optional[pd.DataFrame] = None
    instruments_data: Dict[str, InstrumentData] = None
    benchmark_df: Optional[pd.DataFrame] = None
    benchmark_symbol: str = ""
    trade_start_date: Optional[str] = None
    instrument_close_prices: Dict[str, Dict[str, float]] = None
    compare_data: Dict[str, pd.DataFrame] = None
    turnover: float = 0.0
    total_volume: int = 0
    total_trading_days: int = 0
    pnl_days: tuple = (0, 0)

    def __post_init__(self):
        self.account = self.account or AccountInfo()
        self.config = self.config or BacktestConfig()
        self.strategy_params = self.strategy_params or {}
        self.klines = self.klines or []
        self.trade_log = self.trade_log or []
        self.instruments_data = self.instruments_data or {}
        self.instrument_close_prices = self.instrument_close_prices or {}
        self.compare_data = self.compare_data or {}
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

    def _get_daily_returns(self) -> np.ndarray:
        if self.df is None or self.df.empty or "PortfolioValue" not in self.df.columns:
            return np.array([])
        portfolio_values = self.df["PortfolioValue"].values
        if len(portfolio_values) < 2:
            return np.array([])
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            prev = portfolio_values[i - 1]
            if prev > 0:
                daily_returns.append(portfolio_values[i] / prev - 1)
            else:
                daily_returns.append(0.0)
        return np.array(daily_returns)

    def _get_benchmark_daily_returns(self, benchmark_df: pd.DataFrame) -> np.ndarray:
        if benchmark_df is None or benchmark_df.empty or "PortfolioValue" not in benchmark_df.columns:
            return np.array([])
        values = benchmark_df["PortfolioValue"].values
        if len(values) < 2:
            return np.array([])
        daily_returns = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            if prev > 0:
                daily_returns.append(values[i] / prev - 1)
            else:
                daily_returns.append(0.0)
        return np.array(daily_returns)

    def _align_with_benchmark(self, benchmark_df: pd.DataFrame):
        if self.df is None or self.df.empty or benchmark_df is None or benchmark_df.empty:
            return np.array([]), np.array([])
        strat_returns = self._get_daily_returns()
        bench_returns = self._get_benchmark_daily_returns(benchmark_df)
        min_len = min(len(strat_returns), len(bench_returns))
        if min_len < 2:
            return np.array([]), np.array([])
        return strat_returns[-min_len:], bench_returns[-min_len:]

    def alpha(self, benchmark_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        strat_returns, bench_returns = self._align_with_benchmark(benchmark_df)
        if len(strat_returns) < 2:
            return 0.0
        strat_annual = np.mean(strat_returns) * 252
        bench_annual = np.mean(bench_returns) * 252
        beta_val = self.beta(benchmark_df)
        return float(strat_annual - (risk_free_rate + beta_val * (bench_annual - risk_free_rate)))

    def beta(self, benchmark_df: pd.DataFrame) -> float:
        strat_returns, bench_returns = self._align_with_benchmark(benchmark_df)
        if len(strat_returns) < 2:
            return 0.0
        bench_var = np.var(bench_returns, ddof=1)
        if bench_var == 0:
            return 0.0
        covariance = np.cov(strat_returns, bench_returns, ddof=1)[0, 1]
        return float(covariance / bench_var)

    def tracking_error(self, benchmark_df: pd.DataFrame) -> float:
        strat_returns, bench_returns = self._align_with_benchmark(benchmark_df)
        if len(strat_returns) < 2:
            return 0.0
        active_returns = strat_returns - bench_returns
        te = np.std(active_returns, ddof=1) * np.sqrt(252)
        return float(te)

    def information_ratio(self, benchmark_df: pd.DataFrame) -> float:
        strat_returns, bench_returns = self._align_with_benchmark(benchmark_df)
        if len(strat_returns) < 2:
            return 0.0
        active_returns = strat_returns - bench_returns
        mean_active = np.mean(active_returns) * 252
        te = self.tracking_error(benchmark_df)
        if te == 0:
            return 0.0
        return float(mean_active / te)

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0.0
        excess_returns = daily_returns - risk_free_rate / 252
        downside = excess_returns[excess_returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
        if downside_std == 0:
            return 0.0
        mean_excess = np.mean(excess_returns) * 252
        return float(mean_excess / downside_std)

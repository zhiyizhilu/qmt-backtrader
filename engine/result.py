from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import datetime as dt_module

from core.strategy_logic import StrategyLogic, OrderInfo
from engine.data_feed import ArrayDataFeed
from engine.broker import SimulatedBroker


@dataclass
class EngineResult:
    """自研引擎回测结果容器

    提供与 BacktestingResult 兼容的数据，供 PerformanceAnalyzer 使用。
    """
    equity_history: List[Tuple[dt_module.date, float]] = field(default_factory=list)
    trade_records: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Any] = field(default_factory=list)
    data_feeds: Dict[str, ArrayDataFeed] = field(default_factory=dict)
    broker: Optional[SimulatedBroker] = None
    strategy_logic: Optional[StrategyLogic] = None
    initial_cash: float = 0.0
    final_value: float = 0.0

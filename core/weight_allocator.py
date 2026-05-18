from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np


class WeightAllocator(ABC):
    """权重分配器抽象基类"""

    @abstractmethod
    def allocate(self, symbols: List[str], strategy=None) -> Dict[str, float]:
        """为股票列表分配权重

        Args:
            symbols: 股票代码列表
            strategy: 策略实例，用于获取行情数据

        Returns:
            {symbol: weight} 字典，权重之和为1.0
        """
        pass


class EqualWeightAllocator(WeightAllocator):
    """等权分配器"""

    def allocate(self, symbols: List[str], strategy=None) -> Dict[str, float]:
        if not symbols:
            return {}
        weight = 1.0 / len(symbols)
        return {s: weight for s in symbols}


class RiskParityAllocator(WeightAllocator):
    """风险平价分配器 - 基于历史波动率倒数分配权重"""

    def __init__(self, lookback: int = 60):
        self._lookback = lookback

    def allocate(self, symbols: List[str], strategy=None) -> Dict[str, float]:
        if not symbols:
            return {}

        if len(symbols) == 1:
            return {symbols[0]: 1.0}

        if strategy is None:
            return EqualWeightAllocator().allocate(symbols, strategy)

        inverse_vols = {}
        for symbol in symbols:
            closes = strategy.get_close_prices(symbol, self._lookback)
            if len(closes) < 2:
                inverse_vols[symbol] = None
                continue
            closes_arr = np.array(closes, dtype=np.float64)
            returns = np.diff(closes_arr) / closes_arr[:-1]
            vol = np.std(returns)
            if vol <= 0 or np.isnan(vol):
                inverse_vols[symbol] = None
            else:
                inverse_vols[symbol] = 1.0 / vol

        valid_symbols = [s for s in symbols if inverse_vols[s] is not None]
        if not valid_symbols:
            return EqualWeightAllocator().allocate(symbols, strategy)

        total_inv_vol = sum(inverse_vols[s] for s in valid_symbols)
        if total_inv_vol <= 0:
            return EqualWeightAllocator().allocate(valid_symbols, strategy)

        weights = {}
        for s in valid_symbols:
            weights[s] = inverse_vols[s] / total_inv_vol

        return weights


class FactorWeightAllocator(WeightAllocator):
    """因子加权分配器 - 按因子得分分配权重"""

    def __init__(self, score_fn: Callable[[str], Optional[float]]):
        self._score_fn = score_fn

    def allocate(self, symbols: List[str], strategy=None) -> Dict[str, float]:
        if not symbols:
            return {}

        scores = {}
        for symbol in symbols:
            score = self._score_fn(symbol)
            if score is not None:
                scores[symbol] = score

        if not scores:
            return EqualWeightAllocator().allocate(symbols, strategy)

        total_score = sum(scores.values())
        if total_score <= 0:
            return EqualWeightAllocator().allocate(list(scores.keys()), strategy)

        weights = {}
        for s, score in scores.items():
            weights[s] = score / total_score

        return weights

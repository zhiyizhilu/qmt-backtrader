from typing import Dict, List, Optional
import logging

from core.virtual_book import VirtualBook


class PositionDrift:
    """持仓偏差"""

    def __init__(self, symbol: str, book_volume: int, actual_volume: int):
        self.symbol = symbol
        self.book_volume = book_volume
        self.actual_volume = actual_volume
        self.diff = actual_volume - book_volume

    def __repr__(self):
        return (
            f'PositionDrift({self.symbol}: '
            f'簿记={self.book_volume}, 实际={self.actual_volume}, '
            f'偏差={self.diff})'
        )


class CashDrift:
    """现金偏差"""

    def __init__(self, book_cash: float, actual_cash: float):
        self.book_cash = book_cash
        self.actual_cash = actual_cash
        self.diff = actual_cash - book_cash

    def __repr__(self):
        return (
            f'CashDrift(簿记={self.book_cash:.2f}, '
            f'实际={self.actual_cash:.2f}, 偏差={self.diff:.2f})'
        )


class ReconcileResult:
    """对账结果"""

    def __init__(self):
        self.position_drifts: List[PositionDrift] = []
        self.cash_drift: Optional[CashDrift] = None
        self.is_clean: bool = True

    def __repr__(self):
        if self.is_clean:
            return 'ReconcileResult(一致 ✅)'
        parts = ['ReconcileResult(存在偏差:']
        for d in self.position_drifts:
            parts.append(f'  {d}')
        if self.cash_drift:
            parts.append(f'  {self.cash_drift}')
        parts.append(')')
        return '\n'.join(parts)


class Reconciler:
    """对账器 - 校验虚拟簿记与账户实际状态的一致性

    对账不负责区分策略持仓，VirtualBook 的交易记录才是策略持仓的来源。
    对账只校验"所有策略的簿记之和 = 账户实际"，发现偏差时：
    - 独占标的：自动校准
    - 共享标的：按比例分配并报警
    - 未归属标的：报警
    """

    def __init__(self, virtual_books: List[VirtualBook], qmt_trader):
        """
        Args:
            virtual_books: 同一账户下所有策略的 VirtualBook 列表
            qmt_trader: QMTTrader 实例，用于查询账户实际状态
        """
        self._books = virtual_books
        self._trader = qmt_trader
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def reconcile(self) -> ReconcileResult:
        """执行对账

        Returns:
            ReconcileResult 对账结果
        """
        aggregated_positions = {}
        aggregated_cash = 0.0
        for book in self._books:
            for symbol, volume in book._positions.items():
                aggregated_positions[symbol] = aggregated_positions.get(symbol, 0) + volume
            aggregated_cash += book._cash

        actual_positions = self._query_actual_positions()
        actual_cash = self._query_actual_cash()

        result = ReconcileResult()
        all_symbols = set(aggregated_positions.keys()) | set(actual_positions.keys())

        for symbol in all_symbols:
            book_vol = aggregated_positions.get(symbol, 0)
            actual_vol = actual_positions.get(symbol, 0)
            if book_vol != actual_vol:
                result.position_drifts.append(
                    PositionDrift(symbol, book_vol, actual_vol)
                )
                result.is_clean = False

        if abs(aggregated_cash - actual_cash) > 1.0:
            result.cash_drift = CashDrift(aggregated_cash, actual_cash)
            result.is_clean = False

        return result

    def auto_correct(self, result: ReconcileResult):
        """自动校准偏差

        校准策略：
        - 唯一持有人标的：直接校准到实际值
        - 多持有人标的 + 正偏差（如送股）：按比例分配
        - 多持有人标的 + 负偏差：报警，需人工确认
        - 无持有人但有实际持仓：报警，标记未归属

        Args:
            result: 对账结果
        """
        holders_map = self._build_holders_map()

        for drift in result.position_drifts:
            holders = holders_map.get(drift.symbol, [])

            if len(holders) == 1:
                book = holders[0]
                book._positions[drift.symbol] = drift.actual_volume
                if drift.actual_volume <= 0:
                    book._positions.pop(drift.symbol, None)
                self.logger.info(
                    f'自动校准: {drift.symbol} 偏差{drift.diff}股 '
                    f'→ 归属策略 {book.strategy_id}'
                )

            elif len(holders) > 1 and drift.diff > 0:
                total_book = sum(
                    b._positions.get(drift.symbol, 0) for b in holders
                )
                remaining = drift.diff
                for book in holders:
                    if total_book > 0:
                        ratio = book._positions.get(drift.symbol, 0) / total_book
                        share = int(drift.diff * ratio / 100) * 100
                        if share > 0:
                            book._positions[drift.symbol] = (
                                book._positions.get(drift.symbol, 0) + share
                            )
                            remaining -= share
                if remaining > 0 and holders:
                    holders[0]._positions[drift.symbol] = (
                        holders[0]._positions.get(drift.symbol, 0) + remaining
                    )
                self.logger.warning(
                    f'多策略共享标的 {drift.symbol} 存在偏差: '
                    f'簿记合计={drift.book_volume}, 实际={drift.actual_volume}, '
                    f'已按比例分配'
                )

            elif len(holders) > 1 and drift.diff < 0:
                self.logger.warning(
                    f'多策略共享标的 {drift.symbol} 存在负偏差: '
                    f'簿记合计={drift.book_volume}, 实际={drift.actual_volume}, '
                    f'需人工确认'
                )

            elif len(holders) == 0 and drift.actual_volume > 0:
                self.logger.warning(
                    f'发现未归属持仓: {drift.symbol} {drift.actual_volume}股, 需人工确认'
                )

            else:
                self.logger.warning(
                    f'标的 {drift.symbol} 存在偏差但无法自动校准: '
                    f'簿记={drift.book_volume}, 实际={drift.actual_volume}, 需人工确认'
                )

    def _build_holders_map(self) -> Dict[str, List[VirtualBook]]:
        """构建标的 → 持有人映射"""
        holders_map: Dict[str, List[VirtualBook]] = {}
        for book in self._books:
            for symbol in book._positions:
                holders_map.setdefault(symbol, []).append(book)
        return holders_map

    def _query_actual_positions(self) -> Dict[str, int]:
        """查询账户实际持仓"""
        result: Dict[str, int] = {}
        if not self._trader:
            return result
        try:
            positions = self._trader.get_position()
            if positions:
                for pos in positions:
                    symbol = getattr(pos, 'stock_code', str(pos))
                    volume = getattr(pos, 'volume', 0)
                    if volume > 0:
                        result[symbol] = volume
        except Exception as e:
            self.logger.error(f'查询账户持仓失败: {e}')
        return result

    def _query_actual_cash(self) -> float:
        """查询账户实际现金"""
        if not self._trader:
            return 0.0
        try:
            account = self._trader.get_account()
            if account and hasattr(account, 'cash'):
                return account.cash
        except Exception as e:
            self.logger.error(f'查询账户现金失败: {e}')
        return 0.0

import backtrader as bt
import datetime as dt_module
from typing import Dict, List, Optional, Any
from core.executor import StrategyExecutor, BacktestExecutor
from core.data_adapter import MarketDataAdapter, BacktraderDataAdapter
from core.strategy_logic import StrategyLogic, OrderInfo, TradeInfo, BarData
import logging


class BaseStrategy(bt.Strategy):
    """回测策略适配层 - 将StrategyLogic适配到backtrader框架

    此类不包含任何交易逻辑，仅负责：
    1. 将 backtrader 事件桥接到 StrategyLogic 的事件回调
       - next()       → on_bar()
       - notify_order() → on_order()
       - notify_trade() → on_trade()
    2. 将 backtrader 数据源桥接到数据适配器
    3. 将策略的交易指令桥接到 backtrader 执行引擎
    """

    params = (
        ('param1', 1),
        ('param2', 2),
        ('trade_start_date', None),
    )

    def __init__(self, strategy_logic: Optional[StrategyLogic] = None, **kwargs):
        """初始化回测策略适配层"""
        super().__init__()
        self.dataclose = self.datas[0].close if hasattr(self, 'datas') and self.datas else None
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self._strategy_logic = strategy_logic
        self._equity_history: list = []
        self._trade_records: list = []
        self._last_equity_date = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self._total_bars: int = 0
        self._current_bar: int = 0
        self._progress_bar = None

    def set_strategy_logic(self, strategy_logic: StrategyLogic) -> None:
        """设置策略逻辑实例"""
        self._strategy_logic = strategy_logic

    def get_equity_history(self) -> list:
        """获取权益历史记录

        Returns:
            权益历史列表，每个元素为 (date, value) 元组
        """
        return list(self._equity_history)

    def get_trade_records(self) -> list:
        """获取交易记录

        Returns:
            交易记录列表，每个元素为包含交易详情的字典
        """
        return list(self._trade_records)

    def _get_current_date(self):
        """获取当前回测日期 - 从所有数据源中找到最新的日期

        在多标的策略中，datas[0] 可能在回测期间被退市或停牌，导致其 datetime 冻结。
        此方法遍历所有数据源，返回最新的日期。
        """
        best_date = None
        for data in self.datas:
            try:
                if len(data) > 0:
                    d = data.datetime.date(0)
                    if best_date is None or d > best_date:
                        best_date = d
            except Exception:
                continue
        return best_date or self.datas[0].datetime.date(0)

    def _get_current_datetime(self):
        """获取当前回测时间 - 从所有数据源中找到最新的时间

        与 _get_current_date() 类似，但返回 datetime 对象。
        自动将 00:00 时间调整为 15:00（收盘时间）。
        """
        best_dt = None
        for data in self.datas:
            try:
                if len(data) > 0:
                    d = data.datetime.datetime(0)
                    if best_dt is None or d > best_dt:
                        best_dt = d
            except Exception:
                continue
        if best_dt is None:
            best_dt = self.datas[0].datetime.datetime(0)
        if isinstance(best_dt, dt_module.datetime) and best_dt.hour == 0 and best_dt.minute == 0:
            best_dt = best_dt.replace(hour=15, minute=0)
        return best_dt

    def _init_progress_bar(self):
        """初始化进度条 - 根据数据长度估算总bar数"""
        if not self.datas:
            return
        data = self.datas[0]
        if hasattr(data, 'buflen'):
            self._total_bars = data.buflen()
        elif hasattr(data, 'lines') and len(data.lines) > 0:
            self._total_bars = len(data.lines)
        else:
            self._total_bars = 0

    def next(self):
        self._current_bar += 1
        if self._current_bar == 1:
            self._init_progress_bar()

        if self._strategy_logic:
            self._strategy_logic.update_data()

        dt = self._get_current_date()

        value = self.broker.getvalue()
        if self._last_equity_date != dt:
            self._equity_history.append((dt, value))
            self._last_equity_date = dt
        else:
            if self._equity_history:
                self._equity_history[-1] = (dt, value)

        if self.params.trade_start_date:
            current_date_str = dt.isoformat()
            if current_date_str < self.params.trade_start_date:
                return

        if self._strategy_logic:
            bar = self._build_bar_data()
            self._strategy_logic.on_bar(bar)

        if self._total_bars > 0 and self._current_bar % max(1, self._total_bars // 100) == 0:
            print(f"回测日期: {dt}")


    def _build_bar_data(self) -> BarData:
        """从backtrader数据源构建BarData"""
        if not self.datas:
            return BarData()

        data = self.datas[0]
        bar_datetime = data.datetime.datetime(0)

        return BarData(
            symbol=data._name if hasattr(data, '_name') else '',
            open=data.open[0],
            high=data.high[0],
            low=data.low[0],
            close=data.close[0],
            volume=data.volume[0],
            datetime=bar_datetime,
        )

    def stop(self):
        """回测结束时调用 - 处理未卖出的持仓"""
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

        # 检查是否有未卖出的持仓
        has_position = False
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                has_position = True
                symbol = data._name if hasattr(data, '_name') else 'Unknown'
                size = pos.size
                price = data.close[0]
                value = size * price
                self.log(f'回测结束时仍有持仓: {symbol}, 数量: {size}, 价格: {price:.3f}, 市值: {value:.2f}')

        if has_position:
            # 触发策略的回测结束回调
            if self._strategy_logic and hasattr(self._strategy_logic, 'on_backtest_end'):
                self._strategy_logic.on_backtest_end()

    def notify_order(self, order):
        """backtrader订单通知 - 桥接到 StrategyLogic.on_order()"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        order_info = self._convert_order(order)
        self.order = None

        if self._strategy_logic:
            self._strategy_logic.on_order(order_info)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        trade_info = self._convert_trade(trade)

        dt = self._get_current_datetime() if self.datas else None
        symbol = getattr(trade.data, '_name', '') if hasattr(trade, 'data') else ''
        self._trade_records.append({
            "datetime": dt,
            "symbol": symbol,
            "pnl": trade.pnlcomm if hasattr(trade, 'pnlcomm') else 0.0,
            "pnl_no_commission": trade.pnl if hasattr(trade, 'pnl') else 0.0,
            "price": trade.price if hasattr(trade, 'price') else 0.0,
            "size": int(trade.size) if hasattr(trade, 'size') else 0,
            "commission": trade.commission if hasattr(trade, 'commission') else 0.0,
            "is_long": trade.size > 0 if hasattr(trade, 'size') else True,
        })

        if self._strategy_logic:
            self._strategy_logic.on_trade(trade_info)

    def _convert_order(self, order) -> OrderInfo:
        """将backtrader订单转换为统一的OrderInfo"""
        status_map = {
            order.Submitted: OrderInfo.STATUS_SUBMITTED,
            order.Accepted: OrderInfo.STATUS_ACCEPTED,
            order.Partial: OrderInfo.STATUS_PARTIAL,
            order.Completed: OrderInfo.STATUS_COMPLETED,
            order.Canceled: OrderInfo.STATUS_CANCELED,
            order.Rejected: OrderInfo.STATUS_REJECTED,
            order.Margin: OrderInfo.STATUS_MARGIN,
        }

        direction = 'buy' if order.isbuy() else 'sell'
        status = status_map.get(order.status, '')

        # 获取订单时间
        order_datetime = None
        if hasattr(self, 'datas') and self.datas:
            order_datetime = self._get_current_datetime()

        return OrderInfo(
            order_id=str(id(order)),
            symbol=order.data._name if hasattr(order.data, '_name') else '',
            direction=direction,
            price=order.executed.price if order.status == order.Completed else order.created.price,
            volume=abs(order.created.size),
            status=status,
            executed_volume=abs(order.executed.size) if order.status == order.Completed else 0,
            executed_price=order.executed.price if order.status == order.Completed else 0.0,
            commission=abs(order.executed.comm) if order.status == order.Completed else 0.0,
            datetime=order_datetime,
        )

    def _convert_trade(self, trade) -> TradeInfo:
        """将backtrader交易转换为统一的TradeInfo

        backtrader的Trade对象：
        - trade.size: 平仓后变为0，不能直接使用
        - 需从 trade.history[0]（开仓记录）中获取原始开仓数量
        - trade.history 格式: (status, dt, barlen, size, price, value, pnl, pnlcomm)
        """
        raw_size = 0
        if hasattr(trade, 'history') and len(trade.history) > 0:
            hist = trade.history[0]
            raw_size = hist.get('size', 0) if isinstance(hist, dict) else getattr(hist, 'size', 0)
        if raw_size == 0:
            raw_size = trade.size if hasattr(trade, 'size') else 0

        abs_size = abs(int(raw_size))
        direction = 'buy' if raw_size > 0 else 'sell'

        return TradeInfo(
            trade_id=str(id(trade)),
            order_id='',
            symbol=trade.data._name if hasattr(trade.data, '_name') else '',
            direction=direction,
            price=trade.price,
            volume=abs_size,
            commission=trade.commission,
            pnl=trade.pnl,
        )

    def log(self, txt, dt=None):
        if hasattr(self, 'datas') and self.datas and self.datas[0].datetime:
            dt = dt or self.datas[0].datetime.date(0)
            log_text = f'{dt.isoformat()}, {txt}'
        else:
            log_text = txt
        self.logger.debug(log_text)

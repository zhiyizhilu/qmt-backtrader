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

    def _init_progress_bar(self):
        """初始化进度条 - 根据数据长度估算总bar数"""
        if not self.datas:
            return
        data = self.datas[0]
        if hasattr(data, 'lines') and len(data.lines) > 0:
            self._total_bars = len(data.lines)
        elif hasattr(data, 'buflen'):
            self._total_bars = data.buflen()
        else:
            self._total_bars = 0

    def next(self):
        self._current_bar += 1
        if self._current_bar == 1:
            self._init_progress_bar()

        if self._strategy_logic:
            self._strategy_logic.update_data()

        dt = self.datas[0].datetime.date(0)
        value = self.broker.getvalue()
        if self._last_equity_date != dt:
            self._equity_history.append((dt, value))
            self._last_equity_date = dt
        else:
            if self._equity_history:
                self._equity_history[-1] = (dt, value)

        # 首个bar输出数据状态
        if self._current_bar == 1:
            self.log(f'[DEBUG] 首个bar: 日期={dt}, 数据源数量={len(self.datas)}, 资金={value:.2f}')
            for i, data in enumerate(self.datas):
                name = data._name if hasattr(data, '_name') else f'data[{i}]'
                self.log(f'[DEBUG]   数据源[{i}]: {name}, close={data.close[0]:.3f}, volume={data.volume[0]:.0f}')

        if self.params.trade_start_date:
            current_date_str = dt.isoformat()
            if current_date_str < self.params.trade_start_date:
                # 仅前3个bar输出跳过日志，避免刷屏
                if self._current_bar <= 3:
                    self.log(f'[DEBUG] 跳过(未到交易起始日): 当前={current_date_str}, 交易起始日={self.params.trade_start_date}')
                return

        if self._strategy_logic:
            bar = self._build_bar_data()
            self._strategy_logic.on_bar(bar)

        if self._total_bars > 0 and self._current_bar % max(1, self._total_bars // 100) == 0:
            print(f"[ {self._current_bar} / {self._total_bars} ] 回测日期: {dt}")

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

        dt = self.datas[0].datetime.datetime(0) if self.datas else None
        if isinstance(dt, dt_module.datetime) and dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=14, minute=50)
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
        if hasattr(self, 'datas') and self.datas and self.datas[0].datetime:
            order_datetime = self.datas[0].datetime.datetime(0)
            if isinstance(order_datetime, dt_module.datetime) and order_datetime.hour == 0 and order_datetime.minute == 0:
                order_datetime = order_datetime.replace(hour=14, minute=50)

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
        """将backtrader交易转换为统一的TradeInfo"""
        direction = 'buy' if trade.size > 0 else 'sell'

        return TradeInfo(
            trade_id=str(id(trade)),
            order_id='',
            symbol=trade.data._name if hasattr(trade.data, '_name') else '',
            direction=direction,
            price=trade.price,
            volume=int(trade.size),
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

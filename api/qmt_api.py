import datetime
import time
import threading
from typing import Dict, List, Optional, Any, Callable
import logging
from api.base_api import BaseAPI
from core.executor import QMTExecutor
from core.data_adapter import LiveDataAdapter
from core.data import QMTDataProcessor
from core.strategy_logic import StrategyLogic, BarData


class QMTTrader:
    """QMT交易执行器 - 封装底层交易操作

    职责单一：仅负责与QMT交易接口的交互，
    包括下单、撤单、查询持仓和账户。
    """

    def __init__(self, xttrader=None, xtaccount=None):
        self.xttrader = xttrader
        self.xtaccount = xtaccount
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def buy(self, symbol: str, price: float, volume: int):
        """买入"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return None

        if not symbol or price <= 0 or volume <= 0:
            self.logger.error(f"买入参数无效: symbol={symbol}, price={price}, volume={volume}")
            return None

        try:
            order_id = self.xttrader.order_stock(
                self.xtaccount.account_id,
                symbol,
                price,
                volume,
                'buy',
                'limit'
            )
            self.logger.info(f"买入下单成功，订单号: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"买入下单失败: {e}")
            return None

    def sell(self, symbol: str, price: float, volume: int):
        """卖出"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return None

        if not symbol or price <= 0 or volume <= 0:
            self.logger.error(f"卖出参数无效: symbol={symbol}, price={price}, volume={volume}")
            return None

        try:
            order_id = self.xttrader.order_stock(
                self.xtaccount.account_id,
                symbol,
                price,
                volume,
                'sell',
                'limit'
            )
            self.logger.info(f"卖出下单成功，订单号: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"卖出下单失败: {e}")
            return None

    def cancel_order(self, order_id: str):
        """撤单"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return False

        try:
            result = self.xttrader.cancel_order(self.xtaccount.account_id, order_id)
            self.logger.info(f"撤单成功: {result}")
            return result
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            return False

    def query_order(self, order_id: str):
        """查询订单状态"""
        if not self.xttrader or not self.xtaccount:
            return None
        try:
            orders = self.xttrader.query_orders(self.xtaccount.account_id)
            for order in orders:
                if str(order.order_id) == str(order_id):
                    return order
            return None
        except Exception as e:
            self.logger.error(f"查询订单失败: {e}")
            return None

    def wait_order_completed(self, order_id: str, timeout: float = 30.0, interval: float = 1.0) -> Optional[str]:
        """等待订单完成

        Args:
            order_id: 订单ID
            timeout: 超时时间（秒）
            interval: 轮询间隔（秒）

        Returns:
            订单最终状态，超时返回 None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            order = self.query_order(order_id)
            if order is not None:
                status = getattr(order, 'order_status', None)
                if status in ('filled', 'cancelled', 'rejected'):
                    return status
            time.sleep(interval)
        self.logger.warning(f"等待订单 {order_id} 超时 ({timeout}s)")
        return None

    def get_position(self, symbol: str = None):
        """获取持仓"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return None

        try:
            positions = self.xttrader.query_position(self.xtaccount.account_id)
            if symbol:
                for pos in positions:
                    if pos.stock_code == symbol:
                        return pos
            return positions
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return None

    def get_account(self):
        """获取账户信息"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return None

        try:
            account = self.xttrader.query_account(self.xtaccount.account_id)
            return account
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
            return None


class QMTAPI(BaseAPI):
    """QMT交易API - 编排层，负责初始化QMT连接和驱动策略运行

    交易执行委托给QMTTrader，策略通过QMTExecutor间接调用QMTTrader。

    运行模式：
    - 单次模式: run() 加载历史数据并触发一次 on_bar
    - 持续模式: run_loop() 持续接收行情并定时触发 on_bar
    """

    def __init__(self, is_sim: bool = True, path: str = r'D:\qmt\userdata_mini', account_id: str = None):
        """初始化QMT API"""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.is_sim = is_sim
        self.path = path
        self.account_id = account_id
        self.xttrader = None
        self.xtaccount = None
        self.trader: Optional[QMTTrader] = None
        self.strategy = None
        self.data_processor = QMTDataProcessor()
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._init_api()

    def _init_api(self):
        """初始化API"""
        try:
            import time
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
            from xtquant.xttype import StockAccount
            from xtquant import xtconstant

            # 动态生成 session_id 以避免 -1 连接错误
            session_id = int(time.time() * 1000) % 1000000
            self.xttrader = XtQuantTrader(self.path, session_id)

            # 注册回调以接收交易主推
            class MyXtQuantTraderCallback(XtQuantTraderCallback):
                def __init__(self, api_instance):
                    self.api = api_instance

                def on_disconnected(self):
                    self.api.logger.warning("MiniQMT 交易服务器连接断开")

                def on_stock_order(self, order):
                    self.api.logger.info(f"收到委托主推: {order.stock_code} {order.order_status}")

                def on_stock_trade(self, trade):
                    self.api.logger.info(f"收到成交主推: {trade.stock_code} {trade.traded_volume}@{trade.traded_price}")

            self.callback = MyXtQuantTraderCallback(self)
            self.xttrader.register_callback(self.callback)

            self.xttrader.start()
            connect_result = self.xttrader.connect()
            if connect_result == 0:
                self.logger.info("连接MiniQMT成功")
            else:
                self.logger.error(f"连接MiniQMT失败, 错误码: {connect_result}")

            if self.account_id:
                self.xtaccount = StockAccount(self.account_id)
            else:
                accounts = self.xttrader.query_account_list()
                if accounts:
                    self.xtaccount = accounts[0]
                    self.logger.info(f"自动获取并连接账户成功: {self.xtaccount.account_id}")
                else:
                    self.logger.error("未找到账户")
                    
            if self.xtaccount:
                # 订阅账户以接收主推
                subscribe_result = self.xttrader.subscribe(self.xtaccount)
                if subscribe_result == 0:
                    self.logger.info("订阅账户主推成功")
                else:
                    self.logger.warning(f"订阅账户主推失败, 错误码: {subscribe_result}")

        except ImportError:
            self.logger.error("xtquant 未安装，请从QMT官网下载安装")
        except Exception as e:
            self.logger.error(f"初始化QMT API失败: {e}")

        self.trader = QMTTrader(xttrader=self.xttrader, xtaccount=self.xtaccount)

    def add_strategy(self, strategy_logic_class: type, **kwargs):
        """添加策略 - 实例化StrategyLogic，注入QMT执行器"""
        executor = QMTExecutor(self.trader)
        self.strategy = strategy_logic_class(executor=executor, **kwargs)

    def _load_history_data(self) -> LiveDataAdapter:
        """加载历史数据到数据适配器"""
        adapter = LiveDataAdapter()
        symbols = self.strategy.get_symbols()
        lookback_days = self.strategy.get_lookback_days()

        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

        for symbol in symbols:
            try:
                data = self.data_processor.get_data(symbol, start_date, end_date)
                if not data.empty and 'close' in data.columns:
                    adapter.load_history(symbol, data['close'].tolist())
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")

        adapter.set_current_date(datetime.datetime.now().date())
        return adapter

    def _fetch_realtime_bar(self) -> BarData:
        """从QMT获取实时行情数据构建BarData"""
        symbols = self.strategy.get_symbols()
        now = datetime.datetime.now()

        if symbols:
            symbol = symbols[0]
            try:
                from xtquant import xtdata
                tick = xtdata.get_full_tick([symbol])
                if tick and symbol in tick:
                    t = tick[symbol]
                    return BarData(
                        symbol=symbol,
                        open=getattr(t, 'open', 0),
                        high=getattr(t, 'high', 0),
                        low=getattr(t, 'low', 0),
                        close=getattr(t, 'lastPrice', 0),
                        volume=getattr(t, 'volume', 0),
                        datetime=now,
                    )
            except Exception as e:
                self.logger.warning(f"获取实时行情失败: {e}")

        return BarData(datetime=now)

    def run(self):
        """运行策略 - 单次模式，加载历史数据并触发一次 on_bar"""
        if not self.strategy:
            self.logger.error("未添加策略")
            return

        adapter = self._load_history_data()
        self.strategy.set_data_adapter(adapter)

        bar = BarData(datetime=datetime.datetime.now())
        self.strategy.on_bar(bar)

    def run_loop(self, interval: float = 60.0, on_bar_callback: Optional[Callable] = None):
        """持续运行策略 - 使用订阅回调 (Push) 机制获取行情并触发策略

        Args:
            interval: 轮询间隔（秒），在此模式下仅用于控制保活线程的睡眠间隔
            on_bar_callback: 每次行情到达后的回调函数
        """
        if not self.strategy:
            self.logger.error("未添加策略")
            return

        adapter = self._load_history_data()
        self.strategy.set_data_adapter(adapter)

        self._running = True
        self.logger.info("策略实时数据订阅启动...")

        try:
            from xtquant import xtdata
            symbols = self.strategy.get_symbols()

            def on_data(datas):
                """处理行情推送回调"""
                if not self._running:
                    return
                
                now = datetime.datetime.now()
                # 过滤非交易时间：早于9点，11点半到13点之间，以及15点之后
                hour, minute = now.hour, now.minute
                if hour < 9 or (hour == 11 and minute >= 30) or hour == 12 or hour >= 15:
                    return
                if now.weekday() >= 5:
                    return

                for symbol, data in datas.items():
                    # 兼容不同数据结构的解析
                    close_price = None
                    if isinstance(data, list) and len(data) > 0:
                        close_price = data[-1].get('lastPrice', data[-1].get('close', 0))
                    elif isinstance(data, dict):
                        close_price = data.get('lastPrice', data.get('close', 0))
                    else:
                        try:
                            close_price = getattr(data, 'lastPrice', getattr(data, 'close', 0))
                        except AttributeError:
                            pass
                            
                    if close_price and close_price > 0:
                        if self.strategy._data_adapter:
                            self.strategy._data_adapter.update({
                                symbol: {'close': [close_price]}
                            })
                            self.strategy._data_adapter.set_current_date(now.date())
                            
                        # 构造BarData并触发策略
                        bar = BarData(
                            symbol=symbol,
                            close=close_price,
                            datetime=now
                        )
                        self.strategy.on_bar(bar)

                if on_bar_callback:
                    on_bar_callback(self.strategy)

            for symbol in symbols:
                # 订阅实时行情，设置 callback 以 Push 模式接收数据
                xtdata.subscribe_quote(symbol, period='tick', count=-1, callback=on_data)

            self.logger.info(f"已成功订阅 {len(symbols)} 个标的的实时行情")

            # 保活线程，维持 _running 状态
            def _loop():
                while self._running:
                    time.sleep(interval)
                self.logger.info("策略运行已停止")

            self._loop_thread = threading.Thread(target=_loop, daemon=True)
            self._loop_thread.start()

        except Exception as e:
            self.logger.error(f"实时行情订阅异常: {e}")

    def stop_loop(self):
        """停止策略循环"""
        self._running = False
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)
        self.logger.info("策略循环停止请求已发送")

    def close(self):
        """关闭API"""
        self.stop_loop()
        if self.xttrader:
            try:
                self.xttrader.stop()
                self.logger.info("关闭QMT API成功")
            except Exception as e:
                self.logger.error(f"关闭QMT API失败: {e}")

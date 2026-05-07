import datetime
import time
import threading
from typing import Dict, List, Optional, Any, Callable
import logging
from api.base_api import BaseAPI
from core.executor import QMTExecutor
from core.data_adapter import LiveDataAdapter, QMTLiveDataAdapter, get_trade_unit, validate_stock_code, validate_trade_volume
from core.data import QMTDataProcessor
from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from core.virtual_book import VirtualBook
from core.order_router import OrderRouter


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

        # 校验股票代码格式
        if not validate_stock_code(symbol):
            self.logger.error(f"股票代码格式无效: {symbol}, 正确格式如 000001.SZ")
            return None

        # 校验交易数量
        is_valid, err_msg = validate_trade_volume(symbol, volume)
        if not is_valid:
            self.logger.error(f"买入{err_msg}: symbol={symbol}, volume={volume}")
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

        # 校验股票代码格式
        if not validate_stock_code(symbol):
            self.logger.error(f"股票代码格式无效: {symbol}, 正确格式如 000001.SZ")
            return None

        # 卖出数量无严格限制（允许零股卖出），但仍做基本校验
        # 注意：科创板持仓不足200股时需一次性卖完，这里不做限制由券商校验

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

    @staticmethod
    def get_position_volume(position_obj) -> int:
        """从持仓对象中获取持仓数量（兼容不同版本QMT）"""
        # 优先尝试常见的属性名，避免硬编码依赖
        for attr_name in ('m_nVolume', 'volume', 'total_volume', 'total_quantity'):
            if hasattr(position_obj, attr_name):
                return getattr(position_obj, attr_name, 0)
        return 0


class QMTAPI(BaseAPI):
    """QMT交易API - 编排层，负责初始化QMT连接和驱动策略运行

    交易执行委托给QMTTrader，策略通过QMTExecutor间接调用QMTTrader。

    支持两种模式：
    - 单策略模式：add_strategy() 不传 instance_id，兼容旧用法
    - 多策略模式：add_strategy() 传入 instance_id + virtual_book，支持同账户多策略

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
        self.order_router = OrderRouter()
        self._strategies: Dict[str, StrategyLogic] = {}
        self._executors: Dict[str, QMTExecutor] = {}
        self._virtual_books: Dict[str, VirtualBook] = {}
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
                    self.api._on_qmt_order(order)

                def on_stock_trade(self, trade):
                    self.api.logger.info(f"收到成交主推: {trade.stock_code} {trade.traded_volume}@{trade.traded_price}")
                    self.api._on_qmt_trade(trade)

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

    def add_strategy(self, strategy_logic_class: type, instance_id: str = None,
                     virtual_book: VirtualBook = None, **kwargs):
        """添加策略 - 实例化StrategyLogic，注入QMT执行器

        Args:
            strategy_logic_class: 策略逻辑类
            instance_id: 策略实例ID，用于多策略模式下的订单路由
            virtual_book: 虚拟持仓簿，传入后持仓/资金查询走簿记
            **kwargs: 策略参数
        """
        executor = QMTExecutor(self.trader, virtual_book=virtual_book)

        if instance_id:
            executor.set_order_router(self.order_router)
            strategy = strategy_logic_class(executor=executor, **kwargs)
            self._strategies[instance_id] = strategy
            self._executors[instance_id] = executor
            if virtual_book:
                self._virtual_books[instance_id] = virtual_book
            self.order_router.register_instance(instance_id, strategy)
            self.strategy = strategy
        else:
            self.strategy = strategy_logic_class(executor=executor, **kwargs)

    def _on_qmt_order(self, qmt_order):
        """将QMT委托主推桥接到策略的 on_order 回调

        优先通过 OrderRouter 路由到对应策略实例，
        如果路由失败则回退到单策略模式直接推送。
        """
        try:
            from xtquant import xtconstant

            status_raw = getattr(qmt_order, 'order_status', None)
            status = self._map_qmt_order_status(status_raw)

            direction = 'buy' if getattr(qmt_order, 'order_side', '') == xtconstant.STOCK_BUY else 'sell'

            order_info = OrderInfo(
                order_id=str(getattr(qmt_order, 'order_id', '')),
                symbol=getattr(qmt_order, 'stock_code', ''),
                direction=direction,
                price=float(getattr(qmt_order, 'price', 0)),
                volume=int(getattr(qmt_order, 'order_volume', 0)),
                status=status,
                executed_volume=int(getattr(qmt_order, 'traded_volume', 0)),
                executed_price=float(getattr(qmt_order, 'traded_price', 0)),
            )

            order_id = order_info.order_id
            if self.order_router.has_order(order_id):
                self.order_router.route_order(order_id, order_info)
                if not order_info.is_active:
                    self.order_router.cleanup_order(order_id)
                    if order_info.is_completed or order_info.status == OrderInfo.STATUS_CANCELED:
                        for book in self._virtual_books.values():
                            book.on_order_completed(order_id)
            elif self.strategy:
                self.strategy.on_order(order_info)
        except Exception as e:
            self.logger.error(f'桥接委托回调失败: {e}')

    def _on_qmt_trade(self, qmt_trade):
        """将QMT成交主推桥接到策略的 on_trade 回调

        优先通过 OrderRouter 路由到对应策略实例，
        路由时同步更新 VirtualBook 的簿记。
        """
        try:
            from xtquant import xtconstant

            direction = 'buy' if getattr(qmt_trade, 'order_side', '') == xtconstant.STOCK_BUY else 'sell'

            trade_info = TradeInfo(
                trade_id=str(getattr(qmt_trade, 'traded_id', '')),
                order_id=str(getattr(qmt_trade, 'order_id', '')),
                symbol=getattr(qmt_trade, 'stock_code', ''),
                direction=direction,
                price=float(getattr(qmt_trade, 'traded_price', 0)),
                volume=int(getattr(qmt_trade, 'traded_volume', 0)),
            )

            order_id = trade_info.order_id
            if self.order_router.has_order(order_id):
                instance_id = self.order_router.get_instance_id(order_id)
                self.order_router.route_trade(order_id, trade_info)
                if instance_id and instance_id in self._virtual_books:
                    book = self._virtual_books[instance_id]
                    if trade_info.is_buy:
                        book.on_buy_filled(
                            trade_info.symbol, trade_info.price,
                            trade_info.volume, trade_info.commission
                        )
                    else:
                        book.on_sell_filled(
                            trade_info.symbol, trade_info.price,
                            trade_info.volume, trade_info.commission
                        )
            elif self.strategy:
                self.strategy.on_trade(trade_info)
        except Exception as e:
            self.logger.error(f'桥接成交回调失败: {e}')

    @staticmethod
    def _map_qmt_order_status(status_raw) -> str:
        """将QMT订单状态映射到 OrderInfo 统一状态

        QMT xtconstant 中的订单状态值：
        - MARGIN_CALL, UNKNOWN, INIT, SUBMITTED, ACCEPTED → 活跃
        - FILLED → 完成
        - CANCELED, PARTIAL_CANCEL → 撤单
        - REJECTED → 拒绝
        """
        try:
            from xtquant import xtconstant

            active_statuses = {
                xtconstant.ORDER_MARGIN_CALL,
                xtconstant.ORDER_UNKNOWN,
                xtconstant.ORDER_INIT,
                xtconstant.ORDER_SUBMITTED,
                xtconstant.ORDER_ACCEPTED,
            }
            completed_statuses = {xtconstant.ORDER_FILLED}
            canceled_statuses = {
                xtconstant.ORDER_CANCELED,
                xtconstant.ORDER_PARTIAL_CANCEL,
            }
            rejected_statuses = {xtconstant.ORDER_REJECTED}

            if status_raw in active_statuses:
                return OrderInfo.STATUS_ACCEPTED
            elif status_raw in completed_statuses:
                return OrderInfo.STATUS_COMPLETED
            elif status_raw in canceled_statuses:
                return OrderInfo.STATUS_CANCELED
            elif status_raw in rejected_statuses:
                return OrderInfo.STATUS_REJECTED
        except (ImportError, Exception):
            pass

        if isinstance(status_raw, str):
            status_lower = status_raw.lower()
            if status_lower in ('filled',):
                return OrderInfo.STATUS_COMPLETED
            elif status_lower in ('canceled', 'cancelled', 'partial_cancel'):
                return OrderInfo.STATUS_CANCELED
            elif status_lower in ('rejected',):
                return OrderInfo.STATUS_REJECTED

        return OrderInfo.STATUS_SUBMITTED

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
        """持续运行策略 - 券商版适配：只订阅持仓股，选股按需获取

        改动要点：
        1. 不再 subscribe_quote 订阅全部股票池
        2. 只 subscribe_quote 订阅当前持仓股（10~15只，远小于100只限制）
        3. 使用 QMTLiveDataAdapter 替代 LiveDataAdapter
        4. 选股时通过 get_full_tick / get_market_data_ex 按需获取

        Args:
            interval: 轮询间隔（秒），控制保活线程的睡眠间隔和动态订阅检查频率
            on_bar_callback: 每次行情到达后的回调函数
        """
        if not self.strategy:
            self.logger.error("未添加策略")
            return

        # 使用新的 QMTLiveDataAdapter
        adapter = self._create_live_adapter()
        self.strategy.set_data_adapter(adapter)

        self._running = True
        self._running_adapter = adapter  # 保存引用，供动态订阅使用

        try:
            from xtquant import xtdata

            # 只订阅当前持仓股
            holding_symbols = self._get_holding_symbols()
            if not holding_symbols:
                holding_symbols = ['000300.SH']
                self.logger.info("无持仓，订阅000300.SH作为心跳触发")
            self._subscribe_holdings(holding_symbols, xtdata, on_bar_callback)

            self.logger.info(f"策略启动: 订阅持仓股 {len(holding_symbols)} 只, "
                             f"选股数据通过 get_full_tick 按需获取")

            # 保活线程：定时检查持仓变化，动态调整订阅
            def _loop():
                while self._running:
                    time.sleep(interval)
                    self._update_subscriptions(xtdata, on_bar_callback)
                self.logger.info("策略运行已停止")

            self._loop_thread = threading.Thread(target=_loop, daemon=True)
            self._loop_thread.start()

        except Exception as e:
            self.logger.error(f"策略启动异常: {e}")

    def _create_live_adapter(self) -> 'QMTLiveDataAdapter':
        """创建 QMTLiveDataAdapter，加载持仓股历史数据"""
        adapter = QMTLiveDataAdapter(self.data_processor)

        # 加载持仓股的历史K线
        holding_symbols = self._get_holding_symbols()
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime('%Y-%m-%d')

        for symbol in holding_symbols:
            try:
                data = self.data_processor.get_data(symbol, start_date, end_date)
                if data is not None and not data.empty and 'close' in data.columns:
                    adapter._kline_cache[symbol] = data['close'].tolist()
            except Exception:
                pass

        # 同时也加载策略 get_symbols() 返回的标的（兼容非选股策略）
        if not holding_symbols:
            symbols = self.strategy.get_symbols()
            for symbol in symbols:
                try:
                    data = self.data_processor.get_data(symbol, start_date, end_date)
                    if data is not None and not data.empty and 'close' in data.columns:
                        adapter._kline_cache[symbol] = data['close'].tolist()
                except Exception:
                    pass

        adapter.set_current_date(datetime.datetime.now().date())
        return adapter

    def _get_holding_symbols(self) -> List[str]:
        """获取当前持仓股票列表"""
        if not self.trader:
            return []
        positions = self.trader.get_position()
        if positions is None:
            return []
        if isinstance(positions, (list, tuple)):
            return [p.stock_code for p in positions if QMTTrader.get_position_volume(p) > 0]
        return []

    def _subscribe_holdings(self, symbols: List[str], xtdata, callback):
        """订阅持仓股的实时行情"""
        # 先反订阅旧的
        if hasattr(self, '_holding_sub_ids'):
            for sub_id in self._holding_sub_ids:
                try:
                    xtdata.unsubscribe_quote(sub_id)
                except Exception:
                    pass

        self._holding_sub_ids = []
        self._holding_symbols = set(symbols)

        # 定义 on_data 回调
        api_self = self

        def on_data(datas):
            """处理行情推送回调 - 只更新持仓股价格"""
            if not api_self._running:
                return

            now = datetime.datetime.now()
            hour, minute = now.hour, now.minute
            if hour < 9 or (hour == 11 and minute >= 30) or hour == 12 or hour >= 15:
                return
            if now.weekday() >= 5:
                return

            adapter = getattr(api_self, '_running_adapter', None)
            if not adapter:
                return

            for symbol, data in datas.items():
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
                    # 更新适配器中的订阅价格缓存
                    adapter.set_subscribed_price(symbol, close_price)

            # 触发策略 on_bar
            bar = BarData(datetime=now)
            api_self.strategy.on_bar(bar)

            if callback:
                callback(api_self.strategy)

        # 订阅持仓股
        for symbol in symbols:
            sub_id = xtdata.subscribe_quote(
                symbol, period='tick', count=0, callback=on_data
            )
            if sub_id > 0:
                self._holding_sub_ids.append(sub_id)
            else:
                self.logger.warning(f"订阅 {symbol} 失败")

        self.logger.info(f"已订阅 {len(self._holding_sub_ids)} 只持仓股")

        # 保存 on_data 引用，供后续更新订阅时复用
        self._on_data_callback = on_data

    def _update_subscriptions(self, xtdata, callback):
        """检查持仓变化，动态调整订阅"""
        current_holdings = set(self._get_holding_symbols())
        previous_holdings = getattr(self, '_holding_symbols', set())

        if not current_holdings:
            current_holdings = {'000300.SH'}

        if current_holdings != previous_holdings:
            new_stocks = current_holdings - previous_holdings
            removed_stocks = previous_holdings - current_holdings

            if new_stocks:
                self.logger.info(f"新增持仓，订阅: {new_stocks}")
            if removed_stocks:
                self.logger.info(f"清仓标的，取消订阅: {removed_stocks}")

            self._subscribe_holdings(list(current_holdings), xtdata, callback)

            if hasattr(self, '_running_adapter'):
                for symbol in removed_stocks:
                    self._running_adapter.remove_subscribed_price(symbol)

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

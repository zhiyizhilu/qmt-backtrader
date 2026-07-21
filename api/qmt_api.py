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
        self.pending_orders = {}  # {order_id: {stock_code, direction, volume, submit_time, retry_count}}
        self._on_retry_callback = None  # 撤单重下回调，由 QMTAPI 设置

    def buy(self, symbol: str, price: float, volume: int, strategy_name: str = '', order_remark: str = ''):
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
            from xtquant import xtconstant
            remark = order_remark or (f"{strategy_name}_buy" if strategy_name else "buy")
            order_id = self.xttrader.order_stock(
                self.xtaccount, symbol, xtconstant.STOCK_BUY,
                volume, xtconstant.FIX_PRICE, price,
                remark, symbol
            )
            if order_id and order_id != -1:
                self.logger.info(f"买入下单成功，订单号: {order_id}")
                self.add_pending_order(order_id, symbol, xtconstant.STOCK_BUY, volume, "买入", strategy_name)
                return order_id
            else:
                self.logger.error(f"买入下单失败，返回: {order_id}")
                return None
        except Exception as e:
            self.logger.error(f"买入下单异常: {e}")
            return None

    def sell(self, symbol: str, price: float, volume: int, strategy_name: str = '', order_remark: str = ''):
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
            from xtquant import xtconstant
            remark = order_remark or (f"{strategy_name}_sell" if strategy_name else "sell")
            order_id = self.xttrader.order_stock(
                self.xtaccount, symbol, xtconstant.STOCK_SELL,
                volume, xtconstant.FIX_PRICE, price,
                remark, symbol
            )
            if order_id and order_id != -1:
                self.logger.info(f"卖出下单成功，订单号: {order_id}")
                self.add_pending_order(order_id, symbol, xtconstant.STOCK_SELL, volume, "卖出", strategy_name)
                return order_id
            else:
                self.logger.error(f"卖出下单失败，返回: {order_id}")
                return None
        except Exception as e:
            self.logger.error(f"卖出下单异常: {e}")
            return None

    def cancel_order(self, order_id: str):
        """撤单"""
        if not self.xttrader or not self.xtaccount:
            self.logger.error("交易接口未初始化")
            return False

        try:
            result = self.xttrader.cancel_order_stock(self.xtaccount, order_id)
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
            orders = self.xttrader.query_stock_orders(self.xtaccount)
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
            positions = self.xttrader.query_stock_positions(self.xtaccount)
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
            account = self.xttrader.query_stock_asset(self.xtaccount)
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

    def add_pending_order(self, order_id, stock_code, direction, volume, order_type_name, strategy_name=''):
        """添加待处理订单到跟踪列表

        Args:
            order_id: 订单ID
            stock_code: 股票代码
            direction: 交易方向 (xtconstant.STOCK_BUY / xtconstant.STOCK_SELL)
            volume: 委托数量
            order_type_name: 订单类型名称（"买入"/"卖出"）
            strategy_name: 策略名称，用于重试时传递 order_remark
        """
        self.pending_orders[order_id] = {
            'stock_code': stock_code,
            'direction': direction,
            'volume': volume,
            'submit_time': time.time(),
            'order_type_name': order_type_name,
            'strategy_name': strategy_name,
            'retry_count': 0,
        }
        self.logger.info(f"添加待处理订单: {order_id}, {stock_code}, {order_type_name}, 数量: {volume}")

    def remove_pending_order(self, order_id):
        """从待处理订单列表中移除订单"""
        if order_id in self.pending_orders:
            info = self.pending_orders.pop(order_id)
            self.logger.info(f"移除待处理订单: {order_id}, {info['stock_code']}, {info['order_type_name']}")

    def calculate_price_limit(self, stock_code, prev_close_price=0):
        """获取股票涨跌停价格

        优先通过 xtdata.get_instrument_detail API 获取，
        失败时基于前收盘价计算（主板 ±10%，创业板/科创板 ±20%）。
        """
        try:
            from xtquant import xtdata
            detail = xtdata.get_instrument_detail(stock_code)
            if detail:
                upper = detail.get('UpStopPrice', 0)
                lower = detail.get('DownStopPrice', 0)
                if upper > 0 and lower > 0:
                    return upper, lower
                if prev_close_price <= 0:
                    prev_close_price = detail.get('PreClose', 0)
        except Exception as e:
            self.logger.warning(f"xtdata获取涨跌停失败: {e}")

        if prev_close_price <= 0:
            return None, None

        # ETF/基金(5/15开头): ±10%, 科创板(688)/创业板(300/301): ±20%, 其他: ±10%
        if stock_code.startswith(('5', '15')):
            ratio = 0.10
        elif stock_code.startswith(('688', '300', '301')):
            ratio = 0.20
        else:
            ratio = 0.10

        upper = round(prev_close_price * (1 + ratio), 2)
        lower = round(prev_close_price * (1 - ratio), 2)
        return upper, lower

    def get_valid_price(self, stock_code, direction, price_to_use=0):
        """获取有效委托价格，确保在涨跌停范围内

        Args:
            stock_code: 股票代码
            direction: 交易方向 (xtconstant.STOCK_BUY / xtconstant.STOCK_SELL)
            price_to_use: 建议价格，0 表示不指定
        """
        from xtquant import xtconstant

        upper, lower = self.calculate_price_limit(stock_code)

        if upper is None or lower is None:
            if price_to_use > 0:
                return price_to_use
            raise ValueError(f"无法获取 {stock_code} 的有效价格")

        # 获取当前最新价
        current_price = 0
        try:
            from xtquant import xtdata
            tick = xtdata.get_full_tick([stock_code])
            if tick and stock_code in tick:
                current_price = getattr(tick[stock_code], 'lastPrice', 0)
        except Exception:
            pass

        if direction == xtconstant.STOCK_BUY:
            if current_price > 0:
                buy_price = round(current_price + 0.01 * 10, 2)
                return min(buy_price, upper)
            elif price_to_use > 0:
                return min(price_to_use, upper)
            return upper
        else:
            if current_price > 0:
                sell_price = round(current_price - 0.01 * 10, 2)
                return max(sell_price, lower)
            elif price_to_use > 0:
                return max(price_to_use, lower)
            return lower

    def check_and_retry_pending_orders(self, retry_timeout_seconds=30, max_retry_count=3):
        """检查待处理订单，对超时未成交的订单进行撤单并重新下单

        仅在09:30之后生效。买入单超时撤单后自动重下，卖出单不自动重下。
        """
        now = datetime.datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            return
        if not self.pending_orders:
            return

        from xtquant import xtconstant
        orders_to_remove = []
        orders_to_retry = []

        for order_id, info in list(self.pending_orders.items()):
            elapsed = time.time() - info['submit_time']
            if elapsed < retry_timeout_seconds:
                continue

            order_obj = self.query_order(order_id)
            if order_obj is None:
                self.logger.warning(f"查询待处理订单 {order_id} 失败，移除跟踪")
                orders_to_remove.append(order_id)
                continue

            final_states = {
                xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
                xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL,
            }
            if getattr(order_obj, 'order_status', None) in final_states:
                self.logger.info(f"待处理订单 {order_id} 已达最终状态，移除跟踪")
                orders_to_remove.append(order_id)
                continue

            stock_code = info['stock_code']
            direction = info['direction']
            volume = info['volume']
            order_type_name = info['order_type_name']
            retry_count = info['retry_count']
            strategy_name = info.get('strategy_name', '')

            self.logger.warning(
                f"{order_type_name}订单 {order_id} ({stock_code}) 已超时 {elapsed:.0f}s 未成交，"
                f"尝试撤单 (重试: {retry_count}/{max_retry_count})"
            )

            cancel_result = self.cancel_order(order_id)
            if cancel_result is not False:
                self.logger.info(f"超时撤单成功，委托ID: {order_id}")
                orders_to_remove.append(order_id)

                if direction == xtconstant.STOCK_SELL:
                    self.logger.info(f"{stock_code} 卖出单撤单后不自动重下，等待下一轮策略触发")
                    continue

                if retry_count >= max_retry_count:
                    self.logger.warning(f"{stock_code} {order_type_name}已达最大重试次数，不再重新下单")
                    continue

                remaining_volume = volume - getattr(order_obj, 'traded_volume', 0)
                if remaining_volume <= 0:
                    continue

                try:
                    new_price = self.get_valid_price(stock_code, direction)
                except ValueError as e:
                    self.logger.error(f"获取 {stock_code} 重试价格失败: {e}，跳过")
                    continue

                order_remark = f"{strategy_name}_retry" if strategy_name else "retry"
                new_order_id = self.xttrader.order_stock(
                    self.xtaccount, stock_code, direction,
                    remaining_volume, xtconstant.FIX_PRICE, new_price,
                    order_remark, stock_code
                )

                if new_order_id and new_order_id != -1 and new_order_id != 0:
                    self.logger.info(f"超时重下成功: {stock_code} {order_type_name} {remaining_volume}股 @ {new_price:.2f}")
                    orders_to_retry.append((new_order_id, stock_code, direction, remaining_volume, order_type_name, strategy_name, retry_count + 1, new_price))
                else:
                    self.logger.error(f"超时重下失败: {stock_code} {order_type_name}")
            else:
                self.logger.error(f"超时撤单失败，委托ID: {order_id}")

        for order_id in orders_to_remove:
            self.pending_orders.pop(order_id, None)

        for new_order_id, stock_code, direction, volume, order_type_name, strategy_name, retry_count, retry_price in orders_to_retry:
            self.pending_orders[new_order_id] = {
                'stock_code': stock_code, 'direction': direction,
                'volume': volume, 'submit_time': time.time(),
                'order_type_name': order_type_name, 'strategy_name': strategy_name,
                'retry_count': retry_count,
            }
            # 通知上层（QMTAPI）注册新订单到 order_router 和 VirtualBook
            if self._on_retry_callback:
                try:
                    self._on_retry_callback(
                        str(new_order_id), stock_code, direction, volume, retry_price, strategy_name
                    )
                except Exception as e:
                    self.logger.error(f'重下回调异常: {e}')


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

    def __init__(self, is_sim: bool = True, path: str = r'D:\qmt\userdata_mini', account_id: str = None,
                 trade_mode: str = 'miniqmt', bridge_url: str = 'http://127.0.0.1:8888'):
        """初始化QMT API

        Args:
            is_sim: 是否模拟模式
            path: miniQMT userdata 路径（仅 miniqmt 模式使用）
            account_id: 账户ID
            trade_mode: 交易模式，'miniqmt'（默认）或 'bridge'
            bridge_url: bridge 模式 server 地址（仅 bridge 模式使用）
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.is_sim = is_sim
        self.path = path
        self.account_id = account_id
        self.trade_mode = trade_mode
        self.bridge_url = bridge_url
        self.xttrader = None
        self.xtaccount = None
        self.trader: Optional[QMTTrader] = None
        self._bridge_callback = None  # bridge 模式回调模拟器
        self.strategy = None
        self.data_processor = QMTDataProcessor()
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._tick_open_prices: Dict[str, float] = {}  # 从tick行情中提取的开盘价缓存
        self.order_router = OrderRouter()
        self._strategies: Dict[str, StrategyLogic] = {}
        self._executors: Dict[str, QMTExecutor] = {}
        self._virtual_books: Dict[str, VirtualBook] = {}
        self._on_trade_filled_callback: Optional[Callable[[str, TradeInfo], None]] = None
        self._init_api()

    def set_on_trade_filled_callback(self, callback):
        """注册成交后状态保存回调

        在 _on_qmt_trade 更新 VirtualBook 簿记后调用，用于触发
        RecordManager 保存策略状态。

        Args:
            callback: 回调函数，签名为 callback(instance_id, trade_info)
        """
        self._on_trade_filled_callback = callback

    def _init_api(self):
        """初始化API - 根据 trade_mode 选择初始化方式"""
        if self.trade_mode == 'bridge':
            self._init_bridge_api()
        else:
            self._init_miniqmt_api()

    def _init_miniqmt_api(self):
        """初始化 miniQMT API（原有逻辑）"""
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

                def on_order_error(self, order_error):
                    self.api.logger.error(
                        f"委托失败: order_id={order_error.order_id}, "
                        f"error_id={order_error.error_id}, "
                        f"error_msg={order_error.error_msg}"
                    )
                    self.api._on_qmt_order_error(order_error)

                def on_cancel_error(self, cancel_error):
                    self.api.logger.error(
                        f"撤单失败: order_id={cancel_error.order_id}, "
                        f"error_id={cancel_error.error_id}, "
                        f"error_msg={cancel_error.error_msg}"
                    )

                def on_account_status(self, status):
                    try:
                        from xtquant import xtconstant
                        status_str = "已连接" if status.status == xtconstant.ACCOUNT_STATUS_OK else \
                                    "已断开" if status.status == xtconstant.ACCOUNT_STATUS_DISCONNECTED else \
                                    "错误" if status.status == xtconstant.ACCOUNT_STATUS_ERROR else "其他"
                        self.api.logger.info(f"账户状态变更: {status.account_id}, 状态: {status_str}")
                    except Exception:
                        self.api.logger.info(f"账户状态变更: {status.account_id}")

            self.callback = MyXtQuantTraderCallback(self)
            self.xttrader.register_callback(self.callback)

            self.xttrader.start()
            connect_result = self.xttrader.connect()
            if connect_result == 0:
                self.logger.info("连接MiniQMT成功")
            else:
                self.logger.error(f"连接MiniQMT失败, 错误码: {connect_result}")

            if self.account_id:
                self.xtaccount = StockAccount(self.account_id, 'STOCK')
            else:
                accounts = self.xttrader.query_account_infos()
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
        # 设置撤单重下回调，确保新订单注册到 order_router 和 VirtualBook
        self.trader._on_retry_callback = self._on_order_retry

    def _init_bridge_api(self):
        """初始化桥接模式 API - 通过 HTTP 调用大 QMT server"""
        try:
            from api.qmt_bridge_client import QMTBridgeClient
            from api.bridge_trader import BridgeTrader

            client = QMTBridgeClient(base_url=self.bridge_url)
            # 验证连接
            version = client.python_version()
            if isinstance(version, dict) and 'error' not in version:
                self.logger.info(f"桥接服务连接成功: {version}")
            else:
                self.logger.error(f"桥接服务连接失败: {version}")

            self.trader = BridgeTrader(
                client=client,
                account_id=self.account_id or 'stock'
            )
            self.logger.info(
                f"桥接模式已初始化: {self.bridge_url}, "
                f"账户={self.account_id or 'stock'}"
            )
        except ImportError as e:
            self.logger.error(f"桥接模式依赖缺失: {e}")
        except Exception as e:
            self.logger.error(f"初始化桥接 API 失败: {e}")

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
            # bridge 模式：为首个实例创建回调模拟器（轮询替代主推）
            if self.trade_mode == 'bridge' and not self._bridge_callback:
                from api.bridge_callback import BridgeCallbackSimulator
                self._bridge_callback = BridgeCallbackSimulator(
                    self, instance_id
                )
                self.logger.info(
                    f"桥接回调模拟器已创建: instance={instance_id}"
                )
        else:
            strategy = strategy_logic_class(executor=executor, **kwargs)

        # 保留 self.strategy 兼容单策略模式，多策略模式下指向最后添加的
        self.strategy = strategy

    def _on_qmt_order(self, qmt_order):
        """将QMT委托主推桥接到策略的 on_order 回调

        优先通过 OrderRouter 路由到对应策略实例，
        如果路由失败则回退到单策略模式直接推送。
        """
        try:
            status_raw = getattr(qmt_order, 'order_status', None)
            status = self._map_qmt_order_status(status_raw)

            # STOCK_BUY=23 (xtconstant.STOCK_BUY)，兼容 miniqmt 和 bridge 模式
            # bridge 模式下 _make_order_obj 已将 m_nDirection(48) 转换为 STOCK_BUY(23)
            direction = 'buy' if getattr(qmt_order, 'order_type', 0) == 23 else 'sell'

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
            if not order_info.is_active:
                self.trader.remove_pending_order(order_id)
            if self.order_router.has_order(order_id):
                self.order_router.route_order(order_id, order_info)
                if not order_info.is_active:
                    self.order_router.cleanup_order(order_id)
                    if order_info.is_completed or order_info.status == OrderInfo.STATUS_CANCELED:
                        for book in self._virtual_books.values():
                            book.on_order_completed(order_id)
            elif self._strategies:
                # 多策略模式：不属于任何实例的订单（外部策略），忽略
                self.logger.debug(f'忽略外部订单: {order_id} {order_info.symbol}')
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
            # STOCK_BUY=23 (xtconstant.STOCK_BUY)，兼容 miniqmt 和 bridge 模式
            direction = 'buy' if getattr(qmt_trade, 'order_type', 0) == 23 else 'sell'

            price = float(getattr(qmt_trade, 'traded_price', 0))
            volume = int(getattr(qmt_trade, 'traded_volume', 0))
            # 手续费：优先使用成交回报中的佣金；若缺失或为0，按成交金额万分之一兜底计算
            raw_commission = float(
                getattr(qmt_trade, 'commission', 0)
                or getattr(qmt_trade, 'm_dCommission', 0)
                or 0
            )
            amount = price * volume
            commission = raw_commission if raw_commission > 0 else amount * 0.0001
            trade_info = TradeInfo(
                trade_id=str(getattr(qmt_trade, 'traded_id', '')),
                order_id=str(getattr(qmt_trade, 'order_id', '')),
                symbol=getattr(qmt_trade, 'stock_code', ''),
                direction=direction,
                price=price,
                volume=volume,
                commission=commission,
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
                    # 成交后触发状态保存回调
                    if self._on_trade_filled_callback and instance_id:
                        try:
                            self._on_trade_filled_callback(instance_id, trade_info)
                        except Exception as cb_e:
                            self.logger.error(f'成交保存回调异常: {cb_e}')
            elif self._strategies:
                # 多策略模式：不属于任何实例的成交（外部策略），忽略
                self.logger.debug(f'忽略外部成交: {order_id} {trade_info.symbol}')
            elif self.strategy:
                self.strategy.on_trade(trade_info)
        except Exception as e:
            self.logger.error(f'桥接成交回调失败: {e}')

    def _on_order_retry(self, new_order_id: str, symbol: str, direction, volume: int, price: float, strategy_name: str):
        """撤单重下回调 - 将新订单注册到 order_router 和 VirtualBook

        QMTTrader 撤单重下后调用此方法，确保新订单号的成交回报
        能正确路由到策略实例并更新 VirtualBook。

        Args:
            new_order_id: 新订单ID
            symbol: 标的代码
            direction: 交易方向 (xtconstant)
            volume: 委托数量
            price: 重下价格
            strategy_name: 策略实例ID (来自 order_remark)
        """
        from xtquant import xtconstant
        is_buy = (direction == xtconstant.STOCK_BUY)
        instance_id = strategy_name  # strategy_name 实际存储的是 instance_id

        # 注册到 order_router
        if instance_id and self.order_router:
            self.order_router.register_order(new_order_id, instance_id)
            self.logger.info(f'重下订单注册: {new_order_id} → {instance_id}, {symbol} {"买入" if is_buy else "卖出"}')

        # 更新 VirtualBook 待确认订单
        if instance_id and instance_id in self._virtual_books:
            book = self._virtual_books[instance_id]
            if is_buy:
                book.on_buy_submitted(symbol, price, volume, new_order_id)
            else:
                book.on_sell_submitted(symbol, price, volume, new_order_id)

    def _on_qmt_order_error(self, order_error):
        """处理委托失败回报 - 从待处理订单中移除并通知策略"""
        order_id = getattr(order_error, 'order_id', None)
        if order_id and hasattr(self.trader, 'remove_pending_order'):
            self.trader.remove_pending_order(order_id)
        if self.strategy and hasattr(self.strategy, 'on_order_error'):
            try:
                self.strategy.on_order_error(order_error)
            except Exception as e:
                self.logger.error(f'策略处理委托失败回调异常: {e}')

    @staticmethod
    def _map_qmt_order_status(status_raw) -> str:
        """将QMT订单状态映射到 OrderInfo 统一状态

        xtquant 实际存在的常量:
        - ORDER_UNREPORTED (48), ORDER_REPORTED (50) → 活跃
        - ORDER_SUCCEEDED (56) → 全部成交
        - ORDER_PART_SUCC, ORDER_PART_CANCEL (53), ORDER_CANCELED (54) → 撤单/部撤
        - ORDER_JUNK (57) → 废单
        """
        try:
            from xtquant import xtconstant

            active_statuses = {
                xtconstant.ORDER_UNREPORTED,
                xtconstant.ORDER_REPORTED,
                xtconstant.ORDER_REPORTED_CANCEL,
            }
            completed_statuses = {xtconstant.ORDER_SUCCEEDED}
            canceled_statuses = {
                xtconstant.ORDER_CANCELED,
                xtconstant.ORDER_PART_CANCEL,
                xtconstant.ORDER_PARTSUCC_CANCEL,
            }
            rejected_statuses = {xtconstant.ORDER_JUNK}

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
            if status_lower in ('filled', 'succeeded'):
                return OrderInfo.STATUS_COMPLETED
            elif status_lower in ('canceled', 'cancelled', 'partial_cancel', 'partsucc_cancel'):
                return OrderInfo.STATUS_CANCELED
            elif status_lower in ('rejected', 'junk'):
                return OrderInfo.STATUS_REJECTED

        return OrderInfo.STATUS_SUBMITTED

    def _load_history_data(self) -> LiveDataAdapter:
        """加载历史数据到数据适配器"""
        adapter = LiveDataAdapter()
        all_symbols = set()
        for inst_strategy in self._strategies.values():
            try:
                all_symbols.update(inst_strategy.get_symbols())
            except Exception:
                pass
        if not all_symbols and self.strategy:
            all_symbols.update(self.strategy.get_symbols())
        lookback_days = 30
        for inst_strategy in self._strategies.values():
            if hasattr(inst_strategy, 'get_lookback_days'):
                lookback_days = max(lookback_days, inst_strategy.get_lookback_days())
        if self.strategy and hasattr(self.strategy, 'get_lookback_days'):
            lookback_days = max(lookback_days, self.strategy.get_lookback_days())

        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

        for symbol in all_symbols:
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
        all_symbols = []
        for inst_strategy in self._strategies.values():
            try:
                all_symbols.extend(inst_strategy.get_symbols())
            except Exception:
                pass
        if not all_symbols and self.strategy:
            all_symbols.extend(self.strategy.get_symbols())
        now = datetime.datetime.now()

        if all_symbols:
            symbol = all_symbols[0]
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
        if not self.strategy and not self._strategies:
            self.logger.error("未添加策略")
            return

        adapter = self._load_history_data()
        for inst_id, inst_strategy in self._strategies.items():
            inst_strategy.set_data_adapter(adapter)
        if not self._strategies and self.strategy:
            self.strategy.set_data_adapter(adapter)

        bar = BarData(datetime=datetime.datetime.now())
        for inst_id, inst_strategy in self._strategies.items():
            inst_strategy.on_bar(bar)
        if not self._strategies and self.strategy:
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
        if not self.strategy and not self._strategies:
            self.logger.error("未添加策略")
            return

        # 使用新的 QMTLiveDataAdapter
        adapter = self._create_live_adapter()
        # 为所有策略实例设置数据适配器
        for inst_id, inst_strategy in self._strategies.items():
            inst_strategy.set_data_adapter(adapter)
        # 兼容单策略模式
        if not self._strategies and self.strategy:
            self.strategy.set_data_adapter(adapter)

        self._running = True
        self._running_adapter = adapter  # 保存引用，供动态订阅使用

        # bridge 模式不依赖 xtdata，走轮询路径
        if self.trade_mode == 'bridge':
            self.logger.info(
                f"策略启动(桥接模式): 回调模拟器轮询间隔={interval}s"
            )

            def _bridge_loop():
                check_count = 0
                while self._running:
                    time.sleep(interval)
                    check_count += 1
                    # 轮询委托/成交，触发回调
                    if self._bridge_callback:
                        try:
                            self._bridge_callback.poll_and_process()
                        except Exception as e:
                            self.logger.error(f"桥接轮询异常: {e}")
                    # 每10个周期输出保活日志
                    if check_count % 10 == 0:
                        pending = (
                            len(self.trader.pending_orders)
                            if self.trader else 0
                        )
                        self.logger.info(
                            f"桥接保活检查 #{check_count} "
                            f"(待处理订单={pending})"
                        )
                self.logger.info("策略运行已停止")

            self._loop_thread = threading.Thread(
                target=_bridge_loop, daemon=True
            )
            self._loop_thread.start()
            return

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
                check_count = 0
                while self._running:
                    time.sleep(interval)
                    check_count += 1
                    self._update_subscriptions(xtdata, on_bar_callback)
                    # 每10个周期输出一次保活日志
                    if check_count % 10 == 0:
                        pending = len(self.trader.pending_orders) if self.trader else 0
                        self.logger.info(
                            f"保活检查 #{check_count} (持仓订阅={len(getattr(self, '_holding_symbols', set()))}只, "
                            f"待处理订单={pending})"
                        )
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
            all_symbols = set()
            for inst_strategy in self._strategies.values():
                try:
                    all_symbols.update(inst_strategy.get_symbols())
                except Exception:
                    pass
            # 兼容单策略模式
            if not all_symbols and self.strategy:
                all_symbols.update(self.strategy.get_symbols())
            for symbol in all_symbols:
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

            # 首次行情回调时输出启动确认
            if not hasattr(api_self, '_first_tick_confirmed') or not api_self._first_tick_confirmed:
                api_self._first_tick_confirmed = True
                api_self.logger.info(f"✅ 首次收到行情推送，策略数据通道已连通 (时间={now.strftime('%H:%M:%S')})")

            adapter = getattr(api_self, '_running_adapter', None)
            if not adapter:
                return

            updated_symbols = []
            for symbol, data in datas.items():
                close_price = None
                open_price = None
                if isinstance(data, list) and len(data) > 0:
                    d = data[-1]
                    close_price = d.get('lastPrice', d.get('close', 0))
                    open_price = d.get('open', 0)
                elif isinstance(data, dict):
                    close_price = data.get('lastPrice', data.get('close', 0))
                    open_price = data.get('open', 0)
                else:
                    try:
                        close_price = getattr(data, 'lastPrice', getattr(data, 'close', 0))
                        open_price = getattr(data, 'open', 0)
                    except AttributeError:
                        pass

                if close_price and close_price > 0:
                    adapter.set_subscribed_price(symbol, close_price)
                    updated_symbols.append(f"{symbol}@{close_price:.3f}")
                    # 缓存开盘价供策略使用（9:30执行时用开盘价成交）
                    if open_price and open_price > 0:
                        api_self._tick_open_prices[symbol] = open_price

            if updated_symbols:
                api_self.logger.debug(f"行情更新: {', '.join(updated_symbols)}")

            api_self.logger.info(f"触发 on_bar (时间={now.strftime('%H:%M:%S')}, 更新{len(updated_symbols)}只)")

            # 触发策略 on_bar，传递首个更新标的的开盘价
            # （策略在9:30触发时需要开盘价来执行交易）
            first_symbol = next(iter(datas.keys()), None)
            first_open = api_self._tick_open_prices.get(first_symbol) if first_symbol else None
            bar = BarData(datetime=now, symbol=first_symbol or '')
            if first_open and first_open > 0:
                bar.open = first_open
                bar.close = first_open  # 开盘时刻close≈open
            # 调用所有策略实例的 on_bar
            for inst_id, inst_strategy in api_self._strategies.items():
                try:
                    inst_strategy.on_bar(bar)
                except Exception as e:
                    api_self.logger.error(f"策略 {inst_id} on_bar 异常: {e}")
            # 兼容单策略模式
            if not api_self._strategies and api_self.strategy:
                api_self.strategy.on_bar(bar)

            # 检查超时订单并重试
            api_self.trader.check_and_retry_pending_orders()

            if callback:
                for inst_id, inst_strategy in api_self._strategies.items():
                    callback(inst_strategy)
                if not api_self._strategies and api_self.strategy:
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

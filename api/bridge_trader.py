# -*- coding: utf-8 -*-
"""桥接模式交易执行器

通过 HTTP 调用大 QMT server 执行交易，实现与 QMTTrader 完全兼容的接口。
返回的持仓/委托/账户对象用 SimpleNamespace 包装，属性访问方式与 xtquant 一致，
确保下游 QMTAPI._on_qmt_order / _on_qmt_trade / get_position 等代码无需修改。

移植自参考项目 utils/bridge_manager.py，适配本项目 QMTTrader 的方法签名。
"""
import time
import logging
import datetime
from typing import Optional
from types import SimpleNamespace

from api.qmt_bridge_client import QMTBridgeClient


# ==================== 本地交易常量（替代 xtquant.xtconstant）====================
# 与 xtquant.xtconstant 实际值一致（已验证）
STOCK_BUY = 23        # xtconstant.STOCK_BUY
STOCK_SELL = 24       # xtconstant.STOCK_SELL
FIX_PRICE = 11        # xtconstant.FIX_PRICE

# 委托方向（QMT 内部 m_nDirection 字段，非 opType）
# 注意：这与 STOCK_BUY/STOCK_SELL 不同，用于解析 get_order_status/get_deal 返回值
DIRECTION_BUY = 48    # QMT 内部买入方向
DIRECTION_SELL = 49   # QMT 内部卖出方向

# 委托状态（与 xtconstant 实际值一致）
ORDER_SUCCEEDED = 56       # 全部成交
ORDER_CANCELED = 54        # 已撤销
ORDER_JUNK = 57            # 废单
ORDER_PART_CANCEL = 53     # 部分成交后撤销
ORDER_PARTSUCC_CANCEL = 52 # 部成待撤
ORDER_UNREPORTED = 48      # 未报
ORDER_REPORTED = 50        # 已报
ORDER_REPORTED_CANCEL = 51 # 已报待撤


def _normalize_stock_code(raw_code):
    """将成交/委托记录中的股票代码标准化为带后缀的格式

    get_order_status/get_deal 返回的 m_strInstrumentID 可能不带交易所后缀
    （如 '600988' 而非 '600988.SH'），需要补全。
    """
    if not raw_code:
        return raw_code
    if '.' in raw_code:
        return raw_code
    if raw_code.startswith(('6', '5', '9')):
        return raw_code + '.SH'
    elif raw_code.startswith(('0', '1', '2', '3')):
        return raw_code + '.SZ'
    return raw_code


def _make_position_obj(pos_dict):
    """将桥接模式返回的持仓 dict 包装为与 miniQMT 兼容的对象

    get_holding 返回字段: StockCode, Volume, OpenPrice, MarketValue, CanUseVolume 等
    包装后属性: stock_code, volume, can_use_volume, open_price, market_value, avg_price
    """
    obj = SimpleNamespace()
    obj.stock_code = pos_dict.get(
        'StockCode', pos_dict.get('m_strInstrumentID', '')
    )
    obj.volume = pos_dict.get('Volume', pos_dict.get('m_nVolume', 0))
    obj.can_use_volume = pos_dict.get(
        'CanUseVolume', pos_dict.get('m_nCanUseVolume', 0)
    )
    obj.open_price = pos_dict.get(
        'OpenPrice', pos_dict.get('m_dAvgPrice', 0)
    )
    obj.market_value = pos_dict.get(
        'MarketValue', pos_dict.get('m_dMarketValue', 0)
    )
    obj.avg_price = obj.open_price
    return obj


def _make_order_obj(order_dict):
    """将桥接模式返回的委托 dict 包装为与 miniQMT 兼容的对象

    get_order_status 返回字段: m_strOrderRef, m_strInstrumentID, m_nEntrustStatus,
    m_nDirection, m_dLimitPrice, m_dTradedPrice, m_nVolumeTotalOriginal 等

    注意：m_nDirection 是 48/49（委托方向），而 _on_qmt_order 中判断方向用的是
    xtconstant.STOCK_BUY(23)/STOCK_SELL(24)。这里需要将 48/49 转换为 23/24，
    以兼容现有的方向判断逻辑。
    """
    obj = SimpleNamespace()
    obj.order_id = str(order_dict.get(
        'm_strOrderSysID', order_dict.get('m_strOrderRef', '')
    ))
    raw_code = order_dict.get('m_strInstrumentID', '')
    obj.stock_code = _normalize_stock_code(raw_code)
    obj.order_status = order_dict.get('m_nEntrustStatus', 0)
    # 方向转换：QMT 内部 48/49 → xtconstant 23/24
    direction_raw = order_dict.get(
        'm_nDirection', order_dict.get('m_eEntrustType', 0)
    )
    if direction_raw == DIRECTION_BUY:
        obj.order_type = STOCK_BUY
    elif direction_raw == DIRECTION_SELL:
        obj.order_type = STOCK_SELL
    else:
        obj.order_type = direction_raw
    obj.order_volume = order_dict.get('m_nVolumeTotalOriginal', 0)
    obj.price = order_dict.get('m_dLimitPrice', order_dict.get('m_dPrice', 0))
    obj.traded_volume = order_dict.get('m_nVolumeTraded', 0)
    obj.traded_price = order_dict.get('m_dTradedPrice', 0)
    obj.status_msg = str(obj.order_status)
    obj.order_remark = order_dict.get('m_strRemark', '')
    obj.custom_tag = ''
    return obj


def _make_trade_obj(deal_dict):
    """将桥接模式返回的成交 dict 包装为与 miniQMT 兼容的对象

    get_deal 返回字段: m_strInstrumentID, m_nDirection, m_dPrice, m_nVolume,
    m_strTradeID, m_strOrderRef, m_strOrderSysID, m_dCommission 等
    """
    obj = SimpleNamespace()
    obj.trade_id = str(deal_dict.get('m_strTradeID', ''))
    obj.order_id = str(deal_dict.get(
        'm_strOrderSysID', deal_dict.get('m_strOrderRef', '')
    ))
    raw_code = deal_dict.get('m_strInstrumentID', '')
    obj.stock_code = _normalize_stock_code(raw_code)
    # 方向转换：QMT 内部 48/49 → xtconstant 23/24
    direction_raw = deal_dict.get('m_nDirection', 0)
    if direction_raw == DIRECTION_BUY:
        obj.order_type = STOCK_BUY
    elif direction_raw == DIRECTION_SELL:
        obj.order_type = STOCK_SELL
    else:
        obj.order_type = direction_raw
    obj.traded_price = deal_dict.get('m_dPrice', 0)
    obj.traded_volume = deal_dict.get('m_nVolume', 0)
    obj.traded_amount = deal_dict.get('m_dTradeAmount', 0)
    obj.commission = deal_dict.get('m_dCommission', 0)
    return obj


class BridgeTrader:
    """桥接模式交易执行器 - 通过 HTTP 调用大 QMT server

    实现与 QMTTrader 完全兼容的接口，内部使用 QMTBridgeClient。
    返回的持仓/委托/账户对象用 SimpleNamespace 包装，
    属性访问方式与 xtquant 一致，确保下游代码无需修改。
    """

    def __init__(self, client: QMTBridgeClient, account_id: str = 'stock'):
        """初始化桥接交易执行器

        Args:
            client: QMTBridgeClient 实例
            account_id: 账户ID或账户类型，默认 'stock'
        """
        self.client = client
        self.account_id = account_id
        self.pending_orders = {}  # {order_id: {stock_code, direction, volume, ...}}
        self.logger = logging.getLogger(
            self.__class__.__module__ + '.' + self.__class__.__name__
        )

    def buy(self, symbol: str, price: float, volume: int,
            strategy_name: str = '', order_remark: str = ''):
        """买入

        Args:
            symbol: 股票代码，如 '000001.SZ'
            price: 下单价格
            volume: 下单数量
            strategy_name: 策略名称
            order_remark: 订单备注

        Returns:
            order_ref 作为 order_id，失败返回 None
        """
        if not symbol or price <= 0 or volume <= 0:
            self.logger.error(
                f"买入参数无效: symbol={symbol}, price={price}, volume={volume}"
            )
            return None
        try:
            remark = order_remark or (
                f"{strategy_name}_buy" if strategy_name else "buy"
            )
            result = self.client.buy_stock(
                symbol, price, volume, pr_type=FIX_PRICE,
                strategy_name=remark, reason=''
            )
            if isinstance(result, dict):
                status = result.get('status')
                if status == 'success':
                    order_ref = result.get('order_ref')
                    if order_ref and order_ref != -1:
                        self.logger.info(
                            f"桥接买入下单成功: {symbol}, order_ref={order_ref}"
                        )
                        self.add_pending_order(
                            order_ref, symbol, STOCK_BUY, volume, "买入",
                            strategy_name
                        )
                        return order_ref
                    self.logger.error(
                        f"桥接买入下单成功但 order_ref 无效: {result}"
                    )
                    return None
                elif status == 'warning':
                    self.logger.warning(
                        f"桥接买入下单未产生委托: {result.get('message', '')}"
                    )
                    return None
                else:
                    self.logger.error(
                        f"桥接买入下单失败: {result.get('message', result.get('error', ''))}"
                    )
                    return None
            return None
        except Exception as e:
            self.logger.error(f"桥接买入下单异常: {e}")
            return None

    def sell(self, symbol: str, price: float, volume: int,
             strategy_name: str = '', order_remark: str = ''):
        """卖出

        Args:
            symbol: 股票代码
            price: 下单价格
            volume: 下单数量
            strategy_name: 策略名称
            order_remark: 订单备注

        Returns:
            order_ref 作为 order_id，失败返回 None
        """
        if not symbol or price <= 0 or volume <= 0:
            self.logger.error(
                f"卖出参数无效: symbol={symbol}, price={price}, volume={volume}"
            )
            return None
        try:
            remark = order_remark or (
                f"{strategy_name}_sell" if strategy_name else "sell"
            )
            result = self.client.sell_stock(
                symbol, price, volume, pr_type=FIX_PRICE,
                strategy_name=remark, reason=''
            )
            if isinstance(result, dict):
                status = result.get('status')
                if status == 'success':
                    order_ref = result.get('order_ref')
                    if order_ref and order_ref != -1:
                        self.logger.info(
                            f"桥接卖出下单成功: {symbol}, order_ref={order_ref}"
                        )
                        self.add_pending_order(
                            order_ref, symbol, STOCK_SELL, volume, "卖出",
                            strategy_name
                        )
                        return order_ref
                    self.logger.error(
                        f"桥接卖出下单成功但 order_ref 无效: {result}"
                    )
                    return None
                elif status == 'warning':
                    self.logger.warning(
                        f"桥接卖出下单未产生委托: {result.get('message', '')}"
                    )
                    return None
                else:
                    self.logger.error(
                        f"桥接卖出下单失败: {result.get('message', result.get('error', ''))}"
                    )
                    return None
            return None
        except Exception as e:
            self.logger.error(f"桥接卖出下单异常: {e}")
            return None

    def cancel_order(self, order_id: str):
        """撤单

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        try:
            result = self.client.cancel_order_by_id(str(order_id))
            if isinstance(result, dict) and result.get('status') == 'success':
                self.logger.info(f"桥接撤单成功: {order_id}")
                return True
            self.logger.error(f"桥接撤单失败: {result}")
            return False
        except Exception as e:
            self.logger.error(f"桥接撤单异常: {e}")
            return False

    def query_order(self, order_id: str):
        """查询订单状态

        Returns:
            与 xtquant 兼容的订单对象（SimpleNamespace），失败返回 None
        """
        try:
            result = self.client.get_value_by_order_id(str(order_id))
            if not isinstance(result, dict):
                return None
            data = result.get('data', result.get('order', result))
            if isinstance(data, dict) and data:
                return _make_order_obj(data)
            return None
        except Exception as e:
            self.logger.error(f"桥接查询委托异常: {e}")
            return None

    def get_position(self, symbol: str = None):
        """获取持仓

        Args:
            symbol: 指定股票代码则返回单只持仓对象，None 返回全部持仓列表

        Returns:
            SimpleNamespace 持仓对象或列表，无持仓返回 None 或 []
        """
        try:
            result = self.client.get_holding()
            if not isinstance(result, dict):
                return None if symbol else []
            positions = []
            for code, pos_dict in result.items():
                if isinstance(pos_dict, dict):
                    pos_dict_copy = dict(pos_dict)
                    pos_dict_copy.setdefault('StockCode', code)
                    pos_obj = _make_position_obj(pos_dict_copy)
                    if symbol and pos_obj.stock_code == symbol:
                        return pos_obj
                    positions.append(pos_obj)
            if symbol:
                return None
            return positions
        except Exception as e:
            self.logger.error(f"桥接查询持仓异常: {e}")
            return None if symbol else []

    def get_account(self):
        """获取账户信息

        Returns:
            SimpleNamespace 对象，含 cash(可用资金) 和 total_asset(总资产) 属性；
            失败返回 None
        """
        try:
            total_result = self.client.get_total_money()
            available_result = self.client.get_available_money()
            if not isinstance(total_result, dict) or not isinstance(
                available_result, dict
            ):
                return None
            obj = SimpleNamespace()
            obj.total_asset = total_result.get('total_money', 0)
            obj.cash = available_result.get('available_money', 0)
            obj.frozen_cash = obj.total_asset - obj.cash
            obj.account_id = self.account_id
            return obj
        except Exception as e:
            self.logger.error(f"桥接获取账户信息异常: {e}")
            return None

    @staticmethod
    def get_position_volume(position_obj) -> int:
        """从持仓对象中获取持仓数量（兼容 SimpleNamespace）"""
        for attr_name in ('m_nVolume', 'volume', 'total_volume', 'total_quantity'):
            if hasattr(position_obj, attr_name):
                return getattr(position_obj, attr_name, 0)
        return 0

    def add_pending_order(self, order_id, stock_code, direction, volume,
                          order_type_name, strategy_name=''):
        """添加待处理订单到跟踪列表

        Args:
            order_id: 订单ID
            stock_code: 股票代码
            direction: 交易方向 (STOCK_BUY / STOCK_SELL)
            volume: 委托数量
            order_type_name: 订单类型名称（"买入"/"卖出"）
            strategy_name: 策略名称
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
        self.logger.info(
            f"添加待处理订单: {order_id}, {stock_code}, "
            f"{order_type_name}, 数量: {volume}"
        )

    def remove_pending_order(self, order_id):
        """从待处理订单列表中移除订单"""
        if order_id in self.pending_orders:
            info = self.pending_orders.pop(order_id)
            self.logger.info(
                f"移除待处理订单: {order_id}, "
                f"{info['stock_code']}, {info['order_type_name']}"
            )

    def calculate_price_limit(self, stock_code, prev_close_price=0):
        """获取股票涨跌停价格

        bridge 模式通过 HTTP get_full_tick 获取昨收价，本地计算涨跌停价。
        """
        try:
            result = self.client.get_full_tick(stock_code)
            if isinstance(result, dict):
                tick = result.get(stock_code, result)
                if isinstance(tick, dict):
                    prev_close = tick.get('lastClose', 0)
                    if prev_close > 0:
                        prev_close_price = prev_close
        except Exception as e:
            self.logger.warning(f"桥接获取 {stock_code} tick数据失败: {e}")

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
            direction: 交易方向 (STOCK_BUY / STOCK_SELL)
            price_to_use: 建议价格，0 表示不指定
        """
        upper, lower = self.calculate_price_limit(stock_code)

        if upper is None or lower is None:
            if price_to_use > 0:
                return price_to_use
            raise ValueError(f"无法获取 {stock_code} 的有效价格")

        # 获取当前最新价
        current_price = 0
        try:
            result = self.client.get_full_tick(stock_code)
            if isinstance(result, dict):
                tick = result.get(stock_code, result)
                if isinstance(tick, dict):
                    current_price = tick.get('lastPrice', 0)
        except Exception:
            pass

        if direction == STOCK_BUY:
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

    def check_and_retry_pending_orders(self, retry_timeout_seconds=30,
                                       max_retry_count=3):
        """检查待处理订单，对超时未成交的订单进行撤单并重新下单

        仅在09:30之后生效。买入单超时撤单后自动重下，卖出单不自动重下。
        """
        now = datetime.datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            return
        if not self.pending_orders:
            return

        orders_to_remove = []

        for order_id, info in list(self.pending_orders.items()):
            elapsed = time.time() - info['submit_time']
            if elapsed < retry_timeout_seconds:
                continue

            order_obj = self.query_order(order_id)
            if order_obj is None:
                self.logger.warning(
                    f"查询待处理订单 {order_id} 失败，移除跟踪"
                )
                orders_to_remove.append(order_id)
                continue

            final_states = {
                ORDER_SUCCEEDED, ORDER_CANCELED, ORDER_JUNK, ORDER_PART_CANCEL,
            }
            if getattr(order_obj, 'order_status', None) in final_states:
                self.logger.info(
                    f"待处理订单 {order_id} 已达最终状态，移除跟踪"
                )
                orders_to_remove.append(order_id)
                continue

            stock_code = info['stock_code']
            direction = info['direction']
            volume = info['volume']
            order_type_name = info['order_type_name']
            retry_count = info['retry_count']
            strategy_name = info.get('strategy_name', '')

            self.logger.warning(
                f"{order_type_name}订单 {order_id} ({stock_code}) "
                f"已超时 {elapsed:.0f}s 未成交，尝试撤单 "
                f"(重试: {retry_count}/{max_retry_count})"
            )

            cancel_result = self.cancel_order(order_id)
            if cancel_result:
                self.logger.info(f"超时撤单成功，委托ID: {order_id}")
                orders_to_remove.append(order_id)

                if direction == STOCK_SELL:
                    self.logger.info(
                        f"{stock_code} 卖出单撤单后不自动重下，"
                        f"等待下一轮策略触发"
                    )
                    continue

                if retry_count >= max_retry_count:
                    self.logger.warning(
                        f"{stock_code} {order_type_name}已达最大重试次数，"
                        f"不再重新下单"
                    )
                    continue

                remaining_volume = volume - getattr(
                    order_obj, 'traded_volume', 0
                )
                if remaining_volume <= 0:
                    continue

                try:
                    new_price = self.get_valid_price(stock_code, direction)
                    new_order_id = self.buy(
                        stock_code, new_price, remaining_volume,
                        strategy_name=strategy_name
                    )
                    if new_order_id:
                        self.logger.info(
                            f"重新下单成功: {stock_code}, "
                            f"新订单号: {new_order_id}"
                        )
                except Exception as e:
                    self.logger.error(f"重新下单失败: {stock_code}, {e}")

        for order_id in orders_to_remove:
            self.remove_pending_order(order_id)

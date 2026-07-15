# -*- coding: utf-8 -*-
"""桥接模式回调模拟器

bridge 模式下大 QMT server 无法主动推送委托/成交主推（HTTP 是请求-响应模式），
需通过主动轮询 /api/order/status 和 /api/order/deal 模拟主推回调。

核心设计：
1. 维护 processed_order_refs / processed_deal_ids 集合去重，避免重复处理
2. 持久化到 trading_records/{instance_id}/，防重启重放
3. 轮询到新委托/成交后，构造 SimpleNamespace 对象调用
   api._on_qmt_order / api._on_qmt_trade，复用现有回调入口

移植自参考项目 utils/bridge_callback.py，适配本项目的 QMTAPI 回调接口。
"""
import os
import json
import logging
from typing import TYPE_CHECKING

from api.bridge_trader import (
    _make_order_obj, _make_trade_obj,
    ORDER_SUCCEEDED, ORDER_CANCELED, ORDER_JUNK, ORDER_PART_CANCEL,
)

if TYPE_CHECKING:
    from api.qmt_api import QMTAPI


class BridgeCallbackSimulator:
    """桥接模式回调模拟器 - 轮询替代主推

    在 QMTAPI.run_loop() 的保活线程中定期调用 poll_and_process()，
    轮询委托和成交状态，发现新记录后构造兼容对象调用 _on_qmt_order / _on_qmt_trade。
    """

    def __init__(self, api: 'QMTAPI', instance_id: str,
                 records_dir: str = 'trading_records'):
        """初始化回调模拟器

        Args:
            api: QMTAPI 实例（需有 _on_qmt_order / _on_qmt_trade 方法）
            instance_id: 策略实例ID（用于持久化文件命名）
            records_dir: 记录文件根目录
        """
        self.api = api
        self.instance_id = instance_id
        self.logger = logging.getLogger(
            self.__class__.__module__ + '.' + self.__class__.__name__
        )
        # 去重集合：避免重复处理同一笔委托/成交
        # bridge 的 get_order_status()/get_deal() 返回当日全量历史，
        # 重启后会把已处理过的记录当成"新"的再次处理，需持久化去重
        self.processed_order_refs = set()
        self.processed_deal_ids = set()
        # 持久化文件路径
        records_instance_dir = os.path.join(records_dir, instance_id)
        os.makedirs(records_instance_dir, exist_ok=True)
        self._processed_file = os.path.join(
            records_instance_dir, f'bridge_processed_{instance_id}.json'
        )
        self._load_processed()

    def poll_and_process(self):
        """主循环调用：轮询委托和成交，触发回调

        应在 QMTAPI.run_loop() 的保活线程中定期调用。
        """
        self._poll_orders()
        self._poll_deals()

    def _poll_orders(self):
        """轮询委托状态变化 → 模拟 on_stock_order"""
        try:
            client = self._get_client()
            if not client:
                return
            result = client.get_order_status()
        except Exception as e:
            self.logger.error(f"桥接轮询委托状态异常: {e}")
            return

        if not isinstance(result, dict):
            return

        orders = result.get('orders', result.get('data', []))
        if not isinstance(orders, list):
            return

        added = False
        for order_dict in orders:
            order_ref = str(order_dict.get('m_strOrderRef', ''))
            if not order_ref or order_ref in self.processed_order_refs:
                continue

            # 构造与 xtquant 兼容的订单对象
            order_obj = _make_order_obj(order_dict)
            status = order_obj.order_status

            # 仅对最终状态触发回调（与主推行为一致：活跃状态不重复推）
            if status in (ORDER_SUCCEEDED, ORDER_CANCELED,
                          ORDER_JUNK, ORDER_PART_CANCEL):
                self.logger.info(
                    f"桥接模式：委托 {order_ref} 状态 {status}，"
                    f"{order_obj.stock_code}"
                )
                try:
                    self.api._on_qmt_order(order_obj)
                except Exception as e:
                    self.logger.error(
                        f"桥接委托回调处理异常: {order_ref}, {e}"
                    )
                self.processed_order_refs.add(order_ref)
                added = True

        if added:
            self._save_processed()

    def _poll_deals(self):
        """轮询成交通知 → 模拟 on_stock_trade"""
        try:
            client = self._get_client()
            if not client:
                return
            result = client.get_deal()
        except Exception as e:
            self.logger.error(f"桥接轮询成交状态异常: {e}")
            return

        if not isinstance(result, dict):
            return

        deals = result.get('deals', result.get('data', []))
        if not isinstance(deals, list):
            return

        added = False
        for deal_dict in deals:
            trade_id = str(deal_dict.get('m_strTradeID', ''))
            if not trade_id or trade_id in self.processed_deal_ids:
                continue

            # 构造与 xtquant 兼容的成交对象
            trade_obj = _make_trade_obj(deal_dict)
            self.logger.info(
                f"桥接模式：成交 {trade_id}，{trade_obj.stock_code} "
                f"{trade_obj.traded_volume}@{trade_obj.traded_price}"
            )
            try:
                self.api._on_qmt_trade(trade_obj)
            except Exception as e:
                self.logger.error(
                    f"桥接成交回调处理异常: {trade_id}, {e}"
                )
            self.processed_deal_ids.add(trade_id)
            added = True

        if added:
            self._save_processed()

    def _get_client(self):
        """从 QMTAPI 的 trader 获取 client"""
        trader = getattr(self.api, 'trader', None)
        if trader and hasattr(trader, 'client'):
            return trader.client
        return None

    def _load_processed(self):
        """从磁盘加载已处理过的委托/成交去重集合"""
        if not os.path.exists(self._processed_file):
            return
        try:
            with open(self._processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.processed_order_refs = set(data.get('order_refs', []))
            self.processed_deal_ids = set(data.get('deal_ids', []))
            self.logger.info(
                f"桥接去重集合已加载: "
                f"{len(self.processed_order_refs)} 笔委托, "
                f"{len(self.processed_deal_ids)} 笔成交"
            )
        except Exception as e:
            self.logger.error(f"加载桥接去重集合失败: {e}")

    def _save_processed(self):
        """将已处理过的委托/成交去重集合持久化到磁盘"""
        try:
            data = {
                'order_refs': sorted(self.processed_order_refs),
                'deal_ids': sorted(self.processed_deal_ids),
            }
            tmp_file = self._processed_file + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_file, self._processed_file)
        except Exception as e:
            self.logger.error(f"保存桥接去重集合失败: {e}")
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

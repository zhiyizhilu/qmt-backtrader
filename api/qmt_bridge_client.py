# -*- coding: utf-8 -*-
"""QMT 桥接 HTTP 客户端

调用大 QMT 里运行的 server 服务（tornado HTTP）的 RESTful API 执行交易。
server 端代码直接复用参考项目：
E:\\jupyter notebook\\automatic\\AutoTrade\\qmt\\DdeMainCapital_bridge\\qmt_bridge\\qmt_server.py

移植自参考项目 qmt_bridge/qmt_client.py，仅保留股票交易和查询相关接口。
"""
import requests
import logging


class QMTBridgeClient:
    """QMT 桥接 HTTP 客户端

    通过 HTTP RESTful API 调用大 QMT server 执行交易/查询。
    每个方法的返回值为服务端 JSON 响应体（dict），无统一 'data' 包装层。
    请求失败时返回 {'error': ..., 'status_code': ...}。
    """

    def __init__(self, base_url="http://127.0.0.1:8888", token="123456789"):
        """初始化客户端

        Args:
            base_url: server 服务地址，默认 http://127.0.0.1:8888
            token: 鉴权 token，需与 server 端 TOKEN 一致
        """
        self.base = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json; charset=utf-8"}
        )
        self.session.headers.update({"X-Token": token})
        self.logger = logging.getLogger(
            self.__class__.__module__ + '.' + self.__class__.__name__
        )

    def _req(self, method, path, **kwargs):
        """统一 HTTP 请求方法

        Args:
            method: HTTP 方法 ('GET'/'POST')
            path: API 路径 (如 '/api/holding')
            **kwargs: 传递给 requests.Session.request 的额外参数

        Returns:
            dict: 服务端 JSON 响应体；失败时返回 {'error', 'status_code'}
        """
        url = "{}{}".format(self.base, path)
        try:
            resp = self.session.request(method, url, timeout=10, **kwargs)
            # 强制 utf-8 解码，避免 Windows 下 ISO-8859-1 导致中文乱码
            resp.encoding = 'utf-8'
            try:
                result = resp.json()
                if resp.status_code >= 400 and isinstance(result, dict):
                    result["status_code"] = resp.status_code
                return result
            except ValueError as je:
                if resp.status_code >= 400:
                    return {
                        "error": "HTTP {} - {}".format(
                            resp.status_code, resp.text[:200]
                        ),
                        "status_code": resp.status_code,
                    }
                self.logger.debug(
                    f"JSON解析失败, path={path}, status={resp.status_code}, "
                    f"body={resp.text[:300]}"
                )
                return {
                    "error": "JSON解析失败: {}".format(str(je)),
                    "status_code": resp.status_code,
                    "raw_body": resp.text[:500],
                }
        except requests.RequestException as e:
            resp_text = ''
            if hasattr(e, 'response') and e.response is not None:
                resp_text = e.response.text[:300]
            return {
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', 500),
                "raw_body": resp_text,
            }

    # ==================== 交易接口 ====================

    def buy_stock(self, stock, price, volume, pr_type=11,
                  strategy_name='', reason=''):
        """买入股票

        Args:
            stock: 股票代码，如 '600000.SH'
            price: 下单价格
            volume: 下单数量（股）
            pr_type: 选价类型，默认 11（指定价/模型价）
            strategy_name: 投资备注/策略名称
            reason: 下单原因

        Returns:
            dict: 成功 {'status':'success', 'order_ref':...}；
                  失败 {'status':'error', 'message':...}
        """
        return self._req('POST', '/api/order/buy', json={
            "stock": stock, "price": price, "volume": volume,
            "prType": pr_type, "strategyName": strategy_name, "reason": reason
        })

    def sell_stock(self, stock, price, volume, pr_type=11,
                   strategy_name='', reason=''):
        """卖出股票

        Args:
            stock: 股票代码，如 '600000.SH'
            price: 下单价格
            volume: 下单数量（股）
            pr_type: 选价类型，默认 11（指定价/模型价）
            strategy_name: 投资备注/策略名称
            reason: 下单原因

        Returns:
            dict: 成功 {'status':'success', 'order_ref':...}；
                  失败 {'status':'error', 'message':...}
        """
        return self._req('POST', '/api/order/sell', json={
            "stock": stock, "price": price, "volume": volume,
            "prType": pr_type, "strategyName": strategy_name, "reason": reason
        })

    def cancel_order_by_id(self, order_id, account_type='stock'):
        """根据委托引用号撤单

        Args:
            order_id: 委托引用号/订单ID
            account_type: 账户类型，默认 'stock'

        Returns:
            dict: {'status':'success', 'message':..., 'order_id':...}
        """
        return self._req('POST', '/api/trade/cancel', json={
            "orderId": order_id, "accountType": account_type
        })

    # ==================== 查询接口 ====================

    def get_order_status(self, account='stock'):
        """查询当日委托状态

        Args:
            account: 账户类型，默认 'stock'

        Returns:
            dict: {'orders': [...]}，每个 order 含 m_strOrderRef, m_strInstrumentID,
                  m_nEntrustStatus, m_nDirection(48买/49卖), m_dLimitPrice,
                  m_nVolumeTotalOriginal, m_dTradedPrice, m_nVolumeTraded 等
        """
        return self._req('POST', '/api/order/status', json={"account": account})

    def get_value_by_order_id(self, order_id, account_type='stock',
                              datatype='ORDER'):
        """查询单笔委托

        Args:
            order_id: 委托引用号
            account_type: 账户类型
            datatype: 数据类型，默认 'ORDER'

        Returns:
            dict: 委托详情
        """
        return self._req('POST', '/api/order/value', json={
            "orderId": order_id, "accountType": account_type, "datatype": datatype
        })

    def get_deal(self, account='stock'):
        """查询当日成交记录

        Args:
            account: 账户类型，默认 'stock'

        Returns:
            dict: {'deals': [...]}，每个 deal 含 m_strInstrumentID, m_nDirection(48买/49卖),
                  m_dPrice, m_nVolume, m_strTradeID, m_strOrderRef, m_strOrderSysID,
                  m_dCommission 等
        """
        return self._req('POST', '/api/order/deal', json={"account": account})

    def get_holding(self, account='stock'):
        """查询当前持仓

        Args:
            account: 账户类型，默认 'stock'

        Returns:
            dict: 以股票代码为 key 的持仓字典，每个 value 含 StockCode, Volume,
                  OpenPrice, MarketValue, CanUseVolume 等
        """
        return self._req('POST', '/api/holding', json={"account": account})

    def get_total_money(self, account='stock'):
        """查询总资金

        Returns:
            dict: {'total_money': float}
        """
        return self._req('POST', '/api/money/total', json={"account": account})

    def get_available_money(self, account='stock'):
        """查询可用资金

        Returns:
            dict: {'available_money': float}
        """
        return self._req('POST', '/api/money/available',
                         json={"account": account})

    # ==================== 行情接口 ====================

    def get_full_tick(self, stocks):
        """获取最新全推行情数据

        Args:
            stocks: 股票代码，如 '600000.SH'（多只用逗号分隔）

        Returns:
            dict: 以股票代码为 key 的行情字典，每个 value 含 lastPrice, lastClose,
                  open, high, low, volume, amount 等
        """
        return self._req('POST', '/api/data/full_tick', json={"stocks": stocks})

    # ==================== 系统接口 ====================

    def python_version(self):
        """获取 QMT Python 版本信息（用于连接验证）

        Returns:
            dict: {'python_version': '...'}；连接失败返回 {'error': ...}
        """
        return self._req('GET', '/api/sys/python_version')

    def close(self):
        """关闭 server 服务（谨慎使用）"""
        return self._req('POST', '/api/sys/shutdown')

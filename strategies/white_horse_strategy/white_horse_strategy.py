from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('white_horse',
                   default_kwargs={'max_stocks': 5},
                  backtest_config={'cash': 1000000, 'commission': 0.0013,
                                   'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                   'period': '1d', 'pool': '沪深300',
                                   'data_lookback_days': 400})  # 400自然日≈285交易日，覆盖220日lookback
class WhiteHorseStrategy(StockSelectionStrategy):
    """白马攻防策略 - 根据市场温度动态切换选股模式

    克隆自聚宽文章: https://www.joinquant.com/view/community/detail/50043
    标题：国庆节献礼：实例说明"白马攻防"策略
    作者：蚂蚁量化

    核心思路：根据沪深300指数的市场温度（冷/温/热），采用不同的选股标准：
    - 冷市防守：选择低PB(PB<1)、高现金流质量、稳定盈利的价值股，按ROA/PB排序
    - 温市均衡：选择低PB(PB<1)、中等现金流质量、正增长的均衡股，按ROA/PB排序
    - 热市进攻：选择高PB(PB>3)、成长性强的成长股，按ROA排序

    市场温度判断：
    - 直接使用沪深300指数(000300.SH)收盘价计算
    - market_height = (近5日均值 - N日最低) / (N日最高 - N日最低)
    - cold: height < 0.20
    - hot: height > 0.90
    - warm: 近60日最高/N日最低 > 1.20（默认状态）

    选股逻辑：
    1. 过滤创业板(30)、科创板(68)、北交所(4/8)、ST、停牌、涨跌停
    2. 根据市场温度选择不同筛选条件
    3. 排序选股，取前N只

    调仓规则：
    - 月度调仓，等权重持仓
    - 最多持仓5只股票

    字段映射说明（已适配QMT框架）：
    - JoinQuant valuation.pb_ratio → QMT: price / Pershareindex.s_fa_bps
    - JoinQuant indicator.inc_return → QMT: Pershareindex.du_return_on_equity（扣非ROE,%）
    - JoinQuant indicator.inc_net_profit_year_on_year → QMT: Pershareindex.inc_net_profit_rate（净利润增长率,%）
    - JoinQuant indicator.roa → QMT: indicator.roa近似值，用Pershareindex.adjusted_net_profit/Balance.tot_assets*100
    - JoinQuant cash_flow.subtotal_operate_cash_inflow → QMT: CashFlow.stot_cash_inflows_oper_act
    - JoinQuant indicator.adjusted_profit → QMT: Pershareindex.adjusted_net_profit
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 5),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        # 市场温度参数
        ('lookback_days', 220),  # 市场温度回看天数（与聚宽一致）
        ('recent_days', 5),  # 近期均值天数
        ('short_lookback_days', 60),  # 短期回看天数
        ('cold_threshold', 0.20),  # 冷市阈值
        ('hot_threshold', 0.90),  # 热市阈值
        ('warm_ratio_threshold', 1.20),  # 温市比例阈值
        ('min_lookback_days', 60),  # 最低回看天数（低于此默认温市）
        # 冷市（防守）参数
        ('cold_max_pb', 1.0),
        ('cold_min_cash_quality', 2.0),  # 与聚宽一致：subtotal_operate_cash_inflow/adjusted_profit
        ('cold_min_roe', 1.5),
        ('cold_min_profit_yoy', -15),
        # 温市（均衡）参数
        ('warm_max_pb', 1.0),
        ('warm_min_cash_quality', 1.0),  # 与聚宽一致
        ('warm_min_roe', 2.0),
        ('warm_min_profit_yoy', 0),
        # 热市（进攻）参数
        ('hot_min_pb', 3.0),
        ('hot_min_cash_quality', 0.5),  # 与聚宽一致
        ('hot_min_roe', 3.0),
        ('hot_min_profit_yoy', 20),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._market_temperature = 'warm'

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        if not pool:
            self.log('股票池为空')
            return []

        # 计算市场温度
        self._calc_market_temperature(pool)
        self.log(f'市场温度: {self._market_temperature}')

        # 过滤不可交易股票
        filtered = self._filter_untradeable(pool)
        self.log(f'过滤不可交易: {len(pool)} -> {len(filtered)} 只')

        if not filtered:
            self.log('过滤后无股票')
            return []

        # 根据市场温度选股
        selected = self._select_by_temperature(filtered)
        return selected

    def _calc_market_temperature(self, pool: List[str]):
        """计算市场温度（直接使用沪深300指数收盘价）

        关键逻辑（与聚宽一致）：温度是状态依赖的！
        - 当 height < 0.20 → cold
        - 当 height > 0.90 → hot
        - 当 height 在 [0.20, 0.90] 且 近60日最高/全期最低 > 1.20 → warm
        - 当 height 在 [0.20, 0.90] 且 ratio <= 1.20 → **保持前值**（不默认warm！）
        """
        lookback = self.params.lookback_days
        recent = self.params.recent_days
        short_lookback = self.params.short_lookback_days
        min_lookback = self.params.min_lookback_days

        # 直接使用沪深300指数数据（与聚宽一致）
        index_symbol = '000300.SH'
        closes = self.get_close_prices(index_symbol, lookback + 1)

        if len(closes) < min_lookback:
            self.log(f'指数数据不足({len(closes)}/{min_lookback}条)，保持当前温度({self._market_temperature})')
            return  # 保持当前温度，不强制改为warm

        # 计算沪深300指数的标准化位置（与聚宽逻辑一致）
        actual_len = len(closes)
        use_recent = min(recent, actual_len)
        recent_mean = sum(closes[-use_recent:]) / use_recent
        min_price = min(closes)
        max_price = max(closes)

        if max_price == min_price:
            self.log('指数价格无波动，保持当前温度')
            return  # 保持当前温度

        height = (recent_mean - min_price) / (max_price - min_price)
        self.log(f'市场位置: height={height:.4f}, 指数={index_symbol}, '
                 f'近{use_recent}日均值={recent_mean:.2f}, '
                 f'{lookback}日最低={min_price:.2f}, 最高={max_price:.2f}')

        if height < self.params.cold_threshold:
            self._market_temperature = 'cold'
        elif height > self.params.hot_threshold:
            self._market_temperature = 'hot'
        else:
            # 判断温市：近60日最高/全期最低 > 1.20 → warm
            # 否则保持前值（与聚宽elif逻辑一致）
            use_short = min(short_lookback, actual_len)
            short_max = max(closes[-use_short:])
            short_ratio = short_max / min_price if min_price > 0 else 1.0
            self.log(f'温市判定: 近60日最高={short_max:.2f}, 全期最低={min_price:.2f}, 比率={short_ratio:.4f}, '
                     f'阈值={self.params.warm_ratio_threshold}, 前温度={self._market_temperature}')
            if short_ratio > self.params.warm_ratio_threshold:
                self._market_temperature = 'warm'
            # else: 保持前值！这是聚宽的关键逻辑——不更新为warm

    def _filter_untradeable(self, pool: List[str]) -> List[str]:
        """过滤创业板、科创板、北交所、ST、停牌、涨跌停"""
        result = []
        for stock in pool:
            code = stock.split('.')[0] if '.' in stock else stock
            # 过滤创业板(30)、科创板(68)、北交所(4/8)
            if code.startswith('30') or code.startswith('68') or \
               code.startswith('4') or code.startswith('8'):
                continue
            # 过滤ST
            if self._is_st_stock(stock):
                continue
            # 过滤停牌
            if self.is_suspended(stock):
                continue
            # 过滤涨跌停
            if self.is_limit_up(stock) or self.is_limit_down(stock):
                continue
            result.append(stock)
        return result

    def _is_st_stock(self, stock: str) -> bool:
        """判断是否为ST股票"""
        try:
            from core.stock_lifecycle import get_lifecycle_manager
            mgr = get_lifecycle_manager()
            info = mgr._data.get(stock)
            if info and info.get('name'):
                name = info['name']
                if 'ST' in name or '*' in name or '退' in name:
                    return True
        except Exception:
            pass
        return False

    def _select_by_temperature(self, pool: List[str]) -> List[str]:
        """根据市场温度执行选股"""
        # 批量获取财务数据
        pershare_fields = ['s_fa_bps', 'du_return_on_equity', 'inc_net_profit_rate']
        pershare_data = self.get_financial_fields_batch(pool, 'Pershareindex', pershare_fields)

        # Balance表：使用tot_assets（QMT标准字段名）
        balance_fields = ['tot_assets']
        balance_data = self.get_financial_fields_batch(pool, 'Balance', balance_fields)

        # Income表：归母净利润（QMT adjusted_net_profit恒为0，用net_profit替代）
        income_fields = ['net_profit_incl_min_int_inc_after']
        income_data = self.get_financial_fields_batch(pool, 'Income', income_fields)

        # CashFlow表：经营活动现金流入小计（与聚宽subtotal_operate_cash_inflow一致）
        cashflow_fields = ['stot_cash_inflows_oper_act']
        cashflow_data = self.get_financial_fields_batch(pool, 'CashFlow', cashflow_fields)

        # 计算PB、ROA和现金流质量
        pb_data = {}
        roa_data = {}
        cash_quality_data = {}
        adjusted_profit_data = {}  # 归母净利润（用作adjusted_profit代理）
        data_stats = {'no_price': 0, 'no_pb': 0, 'no_roa': 0, 'no_cash_quality': 0}

        for stock in pool:
            price = self.get_current_price(stock)
            if price is None or price <= 0:
                data_stats['no_price'] += 1
                continue

            # PB = price / bps
            bps = pershare_data.get(stock, {}).get('s_fa_bps')
            if bps and bps > 0:
                pb = price / bps
                pb_data[stock] = pb
            else:
                pb_data[stock] = None
                data_stats['no_pb'] += 1

            # ROA = 归母净利润 / 总资产 * 100
            # 注意：QMT的adjusted_net_profit字段恒为0，不可用
            # 聚宽indicator.roa使用扣非净利润，QMT用归母净利润近似
            net_profit = income_data.get(stock, {}).get('net_profit_incl_min_int_inc_after')
            total_assets = balance_data.get(stock, {}).get('tot_assets')
            if net_profit is not None and total_assets and total_assets > 0:
                roa = net_profit / total_assets * 100
                roa_data[stock] = roa
            else:
                roa_data[stock] = None
                data_stats['no_roa'] += 1

            # 现金流质量 = 经营活动现金流入小计 / 归母净利润
            # 聚宽: subtotal_operate_cash_inflow / adjusted_profit > threshold
            # QMT: stot_cash_inflows_oper_act / net_profit_incl_min_int_inc_after
            # （adjusted_net_profit恒为0，使用net_profit作为代理）
            cash_inflow = cashflow_data.get(stock, {}).get('stot_cash_inflows_oper_act')
            adj_profit = net_profit  # 归母净利润作为扣非净利润代理

            # 存储净利润绝对值（用于后续筛选条件：净利润>0）
            adjusted_profit_data[stock] = adj_profit

            if cash_inflow is not None and cash_inflow > 0 and adj_profit is not None and adj_profit > 0:
                cash_quality = cash_inflow / adj_profit
                cash_quality_data[stock] = cash_quality
            elif cash_inflow is not None and cash_inflow > 0 and adj_profit is not None and adj_profit <= 0:
                # 净利润为负但现金流入为正 → 给一个较高值
                cash_quality_data[stock] = 999.0
            else:
                cash_quality_data[stock] = None
                data_stats['no_cash_quality'] += 1

        # 统计
        has_cash_adj = sum(1 for s in pool if cash_quality_data.get(s) is not None)
        self.log(f'数据统计: {len(pool)}只股票, 无价格={data_stats["no_price"]}, '
                 f'无PB={data_stats["no_pb"]}, 无ROA={data_stats["no_roa"]}, '
                 f'无现金流质量={data_stats["no_cash_quality"]}, '
                 f'有cash_inflow且net_profit>0={has_cash_adj}')

        # 根据温度筛选和排序
        temp = self._market_temperature
        if temp == 'cold':
            return self._select_cold(pool, pb_data, roa_data, cash_quality_data,
                                     pershare_data, adjusted_profit_data)
        elif temp == 'warm':
            return self._select_warm(pool, pb_data, roa_data, cash_quality_data,
                                     pershare_data, adjusted_profit_data)
        else:
            return self._select_hot(pool, pb_data, roa_data, cash_quality_data,
                                    pershare_data, adjusted_profit_data)

    def _select_cold(self, pool: List[str], pb_data: Dict, roa_data: Dict,
                     cash_quality_data: Dict, pershare_data: Dict,
                     adjusted_profit_data: Dict) -> List[str]:
        """冷市防守选股：低PB价值股，按ROA/PB排序"""
        filtered = []
        for stock in pool:
            pb = pb_data.get(stock)
            if pb is None:
                continue
            # PB > 0 and < 1
            if pb <= 0 or pb >= self.params.cold_max_pb:
                continue

            # 现金流质量 > 2.0（与聚宽一致：cash_inflow/adjusted_profit）
            cq = cash_quality_data.get(stock)
            if cq is None:
                continue
            # 扣非净利润必须为正
            adj_profit = adjusted_profit_data.get(stock)
            if adj_profit is None or adj_profit <= 0:
                continue
            if cq < self.params.cold_min_cash_quality:
                continue

            # 扣非ROE > 1.5%
            roe = pershare_data.get(stock, {}).get('du_return_on_equity')
            if roe is None or roe <= self.params.cold_min_roe:
                continue

            # 净利润增长率 > -15%
            profit_yoy = pershare_data.get(stock, {}).get('inc_net_profit_rate')
            if profit_yoy is None or profit_yoy <= self.params.cold_min_profit_yoy:
                continue

            filtered.append(stock)

        # 按ROA/PB排序
        scored = []
        for stock in filtered:
            roa = roa_data.get(stock)
            pb = pb_data.get(stock)
            if roa is not None and pb is not None and pb > 0:
                score = roa / pb
            else:
                score = 0
            scored.append((stock, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [stock for stock, _ in scored[:self.params.max_stocks]]
        self.log(f'冷市选股: {len(pool)} -> {len(filtered)} -> {len(selected)} 只, '
                 f'按ROA/PB排序')
        self._log_top_stocks(scored[:5], pb_data, roa_data, pershare_data, cash_quality_data)
        return selected

    def _select_warm(self, pool: List[str], pb_data: Dict, roa_data: Dict,
                     cash_quality_data: Dict, pershare_data: Dict,
                     adjusted_profit_data: Dict) -> List[str]:
        """温市均衡选股：低PB均衡股，按ROA/PB排序"""
        filtered = []
        for stock in pool:
            pb = pb_data.get(stock)
            if pb is None:
                continue
            # PB > 0 and < 1
            if pb <= 0 or pb >= self.params.warm_max_pb:
                continue

            # 现金流质量 > 1.0（与聚宽一致：cash_inflow/adjusted_profit）
            cq = cash_quality_data.get(stock)
            if cq is None:
                continue
            # 扣非净利润必须为正
            adj_profit = adjusted_profit_data.get(stock)
            if adj_profit is None or adj_profit <= 0:
                continue
            if cq < self.params.warm_min_cash_quality:
                continue

            # 扣非ROE > 2.0%
            roe = pershare_data.get(stock, {}).get('du_return_on_equity')
            if roe is None or roe <= self.params.warm_min_roe:
                continue

            # 净利润增长率 > 0%
            profit_yoy = pershare_data.get(stock, {}).get('inc_net_profit_rate')
            if profit_yoy is None or profit_yoy <= self.params.warm_min_profit_yoy:
                continue

            filtered.append(stock)

        # 按ROA/PB排序
        scored = []
        for stock in filtered:
            roa = roa_data.get(stock)
            pb = pb_data.get(stock)
            if roa is not None and pb is not None and pb > 0:
                score = roa / pb
            else:
                score = 0
            scored.append((stock, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [stock for stock, _ in scored[:self.params.max_stocks]]
        self.log(f'温市选股: {len(pool)} -> {len(filtered)} -> {len(selected)} 只, '
                 f'按ROA/PB排序')
        self._log_top_stocks(scored[:5], pb_data, roa_data, pershare_data, cash_quality_data)
        return selected

    def _select_hot(self, pool: List[str], pb_data: Dict, roa_data: Dict,
                    cash_quality_data: Dict, pershare_data: Dict,
                    adjusted_profit_data: Dict) -> List[str]:
        """热市进攻选股：高PB成长股，按ROA排序"""
        filtered = []
        for stock in pool:
            pb = pb_data.get(stock)
            if pb is None:
                continue
            # PB > 3
            if pb <= self.params.hot_min_pb:
                continue

            # 现金流质量 > 0.5（与聚宽一致：cash_inflow/adjusted_profit）
            cq = cash_quality_data.get(stock)
            if cq is None:
                continue
            # 扣非净利润必须为正
            adj_profit = adjusted_profit_data.get(stock)
            if adj_profit is None or adj_profit <= 0:
                continue
            if cq < self.params.hot_min_cash_quality:
                continue

            # 扣非ROE > 3.0%
            roe = pershare_data.get(stock, {}).get('du_return_on_equity')
            if roe is None or roe <= self.params.hot_min_roe:
                continue

            # 净利润增长率 > 20%
            profit_yoy = pershare_data.get(stock, {}).get('inc_net_profit_rate')
            if profit_yoy is None or profit_yoy <= self.params.hot_min_profit_yoy:
                continue

            filtered.append(stock)

        # 按ROA排序
        scored = []
        for stock in filtered:
            roa = roa_data.get(stock)
            score = roa if roa is not None else 0
            scored.append((stock, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [stock for stock, _ in scored[:self.params.max_stocks]]
        self.log(f'热市选股: {len(pool)} -> {len(filtered)} -> {len(selected)} 只, '
                 f'按ROA排序')
        self._log_top_stocks(scored[:5], pb_data, roa_data, pershare_data, cash_quality_data)
        return selected

    def _log_top_stocks(self, top_stocks: list, pb_data: Dict, roa_data: Dict,
                        pershare_data: Dict, cash_quality_data: Dict):
        """记录排名前N只股票的详细信息"""
        for stock, score in top_stocks:
            pb = pb_data.get(stock)
            roa = roa_data.get(stock)
            roe = pershare_data.get(stock, {}).get('du_return_on_equity')
            profit_yoy = pershare_data.get(stock, {}).get('inc_net_profit_rate')
            adj_profit = pershare_data.get(stock, {}).get('adjusted_net_profit')
            cq = cash_quality_data.get(stock)
            parts = []
            if pb is not None:
                parts.append(f'PB:{pb:.2f}')
            if roa is not None:
                parts.append(f'ROA:{roa:.2f}%')
            if roe is not None:
                parts.append(f'ROE:{roe:.2f}%')
            if profit_yoy is not None:
                parts.append(f'利润增速:{profit_yoy:.2f}%')
            if cq is not None:
                if cq >= 999.0:
                    parts.append(f'现金流质量:inf')
                else:
                    parts.append(f'现金流质量:{cq:.2f}')
            if adj_profit is not None:
                if abs(adj_profit) >= 1e8:
                    parts.append(f'净利润:{adj_profit/1e8:.2f}亿')
                elif abs(adj_profit) >= 1e4:
                    parts.append(f'净利润:{adj_profit/1e4:.2f}万')
                else:
                    parts.append(f'净利润:{adj_profit:.2f}')
            self.log(f'  {stock} | 排名分:{score:.2f} | ' + ' | '.join(parts))

    def on_backtest_end(self):
        pass

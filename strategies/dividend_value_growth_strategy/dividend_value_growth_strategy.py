import datetime as dt_module
from typing import List, Optional
from core.stock_selection import StockSelectionStrategy
from core.data_adapter import get_limit_ratio
from strategies import register_strategy


@register_strategy('dividend_value_growth',
                   default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0013,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指'})
class DividendValueGrowthStrategy(StockSelectionStrategy):
    """高股息低市盈率高增长价投策略 - 稳健穿越牛熊

    克隆自聚宽文章: https://www.joinquant.com/post/45552
    原始策略: 高分红、低市盈率、高增长的价值投资策略

    选股逻辑：
    1. 过滤科创板、北交所、ST、上市未满300天的股票
    2. 按近3年股息率排序，选取前10%的股票
    3. 基本面筛选：PE 0~25，PEG 0.08~1.9，ROE>3%，营收增速>5%，净利润增速>11%
    4. 去停牌、去涨停，取前10只

    调仓规则：
    - 月度调仓，等权重持仓
    - 最多持仓10只股票
    - 每日检查持仓中昨日涨停的股票，若打开涨停则卖出

    已知偏差（vs聚宽原版）：
    - 股息率使用近3次派息均值/当前价格，原版使用近3年累计分红/总市值
    - PE使用价格/稀释EPS计算，原版使用valuation.pe_ratio
    - PEG使用PE/净利润增速，原版使用PE/净利润增速（一致）
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('dividend_top_pct', 0.1),       # 股息率排名前10%
        ('min_pe', 0),
        ('max_pe', 25),
        ('min_peg', 0.08),
        ('max_peg', 1.9),
        ('min_roe', 3),                   # ROE > 3%
        ('min_revenue_growth', 5),        # 营收增速 > 5%
        ('min_profit_growth', 11),        # 净利润增速 > 11%
        ('min_ipo_days', 300),            # 上市天数 >= 300
        # 第二轮优化参数（默认禁用）
        ('sort_by_peg', None),            # B1: 按PEG排序替代股息率排序
        ('min_ocf_ratio', None),          # C1: 经营现金流/净利润最低比率
        ('max_debt_ratio', None),         # C2: 资产负债率上限(%)
        ('min_rebalance_change', None),   # D1: 最小调仓变动股票数
        ('hold_priority', None),          # D2: 持仓续留优先(True/False)
        ('sell_max_pe', None),            # E1: PE超过此值主动卖出
        ('sell_min_dividend_yield', None),# E2: 股息率低于此值主动卖出
        ('use_composite_score', None),    # G1: 使用综合评分排序(True/False)
        ('min_dividend_years', None),     # H1: 最少连续分红年数
        ('max_industry_stocks', None),     # 行业分散: 每行业最多持仓N只
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._high_limit_list: List[str] = []
        self._lifecycle_manager = None

    def _get_lifecycle_manager(self):
        if self._lifecycle_manager is None:
            try:
                from core.stock_lifecycle import get_lifecycle_manager
                self._lifecycle_manager = get_lifecycle_manager()
            except Exception:
                pass
        return self._lifecycle_manager

    def on_bar(self, bar):
        self._prepare_high_limit_list()
        self._check_limit_up()
        self._check_sell_signals()
        super().on_bar(bar)

    def _prepare_high_limit_list(self):
        """检查持仓中昨日涨停的股票"""
        holdings = self.get_current_holdings()
        self._high_limit_list = []

        for symbol in holdings:
            ohlcv = self.get_ohlcv_data(symbol, 3)
            if ohlcv and len(ohlcv) >= 2:
                prev_close = ohlcv[-2].get('close', 0)
                prev_prev_close = None
                if len(ohlcv) >= 3:
                    prev_prev_close = ohlcv[-3].get('close', 0)
                elif len(ohlcv) == 2:
                    closes = self.get_close_prices(symbol, 3)
                    if len(closes) >= 3:
                        prev_prev_close = closes[-3]

                if prev_close > 0 and prev_prev_close and prev_prev_close > 0:
                    limit_ratio = get_limit_ratio(symbol)
                    limit_price = round(prev_prev_close * (1 + limit_ratio), 2)
                    if prev_close >= limit_price - 0.005:
                        self._high_limit_list.append(symbol)

    def _check_limit_up(self):
        """昨日涨停今天开板的卖出"""
        if not self._high_limit_list:
            return

        for symbol in self._high_limit_list[:]:
            if not self.is_limit_up(symbol):
                pos_size = self._current_holdings.get(symbol, 0)
                if pos_size > 0:
                    sellable = self.get_sellable_volume(symbol)
                    if sellable > 0:
                        price = self.get_current_price(symbol)
                        if price and price > 0 and not self.is_limit_down(symbol):
                            self.sell(symbol, price, sellable)
                            if symbol in self._current_holdings:
                                del self._current_holdings[symbol]
                            self.log(f'涨停打开，卖出: {symbol}')

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        # 步骤1: 基本过滤
        filtered = self._filter_kcbj_stock(pool)
        self.log(f'过滤科创北交: {len(pool)} -> {len(filtered)} 只')

        filtered = self._filter_st_stock(filtered)
        self.log(f'过滤ST: -> {len(filtered)} 只')

        filtered = self._filter_new_stock(filtered)
        self.log(f'过滤次新股(<{self.params.min_ipo_days}天): -> {len(filtered)} 只')

        if not filtered:
            self.log('基本过滤后无股票')
            return []

        # 步骤2: 股息率筛选 - 取前10% (含H1分红连续性)
        dividend_stocks = self._filter_by_dividend_yield(filtered)
        self.log(f'股息率前{self.params.dividend_top_pct:.0%}: {len(filtered)} -> {len(dividend_stocks)} 只')

        if not dividend_stocks:
            self.log('股息率筛选后无股票')
            return []

        # 步骤3: 基本面筛选 (PE, PEG, ROE, 营收增速, 净利润增速, 可选C1/C2)
        fundamental_stocks = self._filter_fundamental(dividend_stocks)
        self.log(f'基本面筛选: {len(dividend_stocks)} -> {len(fundamental_stocks)} 只')

        if not fundamental_stocks:
            self.log('基本面筛选后无股票')
            return []

        # 步骤4: 去停牌、去涨停
        tradable = self._filter_tradable(fundamental_stocks)
        self.log(f'去停牌涨停: {len(fundamental_stocks)} -> {len(tradable)} 只')

        # 步骤5: 排序逻辑
        if self.params.use_composite_score:
            # G1: 综合评分排序
            tradable = self._sort_by_composite_score(tradable)
            self.log(f'综合评分排序完成')
        elif self.params.sort_by_peg:
            # B1: 按PEG升序排序
            tradable = self._sort_by_peg(tradable)
            self.log(f'按PEG排序完成')

        # 步骤5b: 行业分散 - 每行业最多持仓N只
        if self.params.max_industry_stocks is not None:
            tradable = self._filter_by_industry_diversity(tradable)

        # 步骤6: D2 持仓续留优先
        max_stocks = self.params.max_stocks
        if self.params.hold_priority:
            holdings = set(self._current_holdings.keys())
            hold_in_pool = [s for s in tradable if s in holdings]
            new_in_pool = [s for s in tradable if s not in holdings]
            selected = hold_in_pool + new_in_pool
            selected = selected[:max_stocks]
            self.log(f'持仓续留优先: 保留{len(hold_in_pool)}只持仓 + 新增{min(len(new_in_pool), max_stocks - len(hold_in_pool))}只')
        else:
            selected = tradable[:max_stocks]

        # 步骤7: D1 最小调仓变动
        if self.params.min_rebalance_change is not None and self._current_holdings:
            current_set = set(self._current_holdings.keys())
            new_set = set(selected)
            to_sell = current_set - new_set
            to_buy = new_set - current_set
            total_changes = len(to_sell) + len(to_buy)
            if total_changes < self.params.min_rebalance_change:
                self.log(f'调仓变动{total_changes}只 < 阈值{self.params.min_rebalance_change}，跳过调仓')
                return list(current_set)

        self.log(f'选股结果: {len(pool)} -> {len(selected)} 只')
        for stock in selected[:10]:
            dy = self.get_dividend_yield(stock)
            pe = self._calc_pe(stock)
            roe = self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')
            parts = [f'股息率:{dy:.2%}' if dy else '股息率:N/A']
            if pe is not None:
                parts.append(f'PE:{pe:.1f}')
            if roe is not None:
                parts.append(f'ROE:{roe:.1f}%')
            self.log(f'  {stock} | ' + ' | '.join(parts))

        return selected

    # ================================================================
    # 过滤方法
    # ================================================================

    def _filter_kcbj_stock(self, stock_list: List[str]) -> List[str]:
        """过滤科创板和北交所股票"""
        result = []
        for stock in stock_list:
            code = stock.split('.')[0] if '.' in stock else stock
            if code.startswith('4') or code.startswith('8') or code.startswith('68'):
                continue
            result.append(stock)
        return result

    def _filter_st_stock(self, stock_list: List[str]) -> List[str]:
        """过滤ST及退市标签股票"""
        lifecycle = self._get_lifecycle_manager()
        if lifecycle is None:
            return stock_list

        result = []
        for stock in stock_list:
            info = lifecycle._data.get(stock)
            if info and info.get('name'):
                name = info['name']
                if 'ST' in name or '*' in name or '退' in name:
                    continue
            result.append(stock)
        return result

    def _filter_new_stock(self, stock_list: List[str]) -> List[str]:
        """过滤上市未满指定天数的次新股"""
        lifecycle = self._get_lifecycle_manager()
        if lifecycle is None:
            return stock_list

        current_date = self.get_current_date()
        if current_date is None:
            return stock_list

        result = []
        for stock in stock_list:
            list_date_str = lifecycle.get_list_date(stock)
            if list_date_str:
                try:
                    list_date = dt_module.date.fromisoformat(list_date_str[:10])
                    if isinstance(current_date, dt_module.datetime):
                        current = current_date.date()
                    elif isinstance(current_date, dt_module.date):
                        current = current_date
                    else:
                        result.append(stock)
                        continue
                    if (current - list_date).days < self.params.min_ipo_days:
                        continue
                except (ValueError, TypeError):
                    pass
            result.append(stock)
        return result

    def _filter_by_dividend_yield(self, stock_list: List[str]) -> List[str]:
        """按股息率排序，选取前N%的股票（含H1分红连续性过滤）"""
        dividend_yields = {}
        for stock in stock_list:
            # H1: 分红连续性
            if self.params.min_dividend_years is not None:
                history = self.get_dvps_history(stock, count=self.params.min_dividend_years + 1)
                if not history or len(history) < self.params.min_dividend_years:
                    continue

            dy = self.get_dividend_yield(stock, use_avg=True)
            if dy is not None and dy > 0:
                dividend_yields[stock] = dy

        if not dividend_yields:
            return []

        sorted_stocks = sorted(dividend_yields.items(), key=lambda x: x[1], reverse=True)
        top_n = max(1, int(len(sorted_stocks) * self.params.dividend_top_pct))
        return [stock for stock, _ in sorted_stocks[:top_n]]

    def _filter_fundamental(self, stock_list: List[str]) -> List[str]:
        """基本面筛选: PE, PEG, ROE, 营收增速, 净利润增速, 可选OCF/负债率"""
        if not stock_list:
            return []

        pershare_fields = ['s_fa_eps_diluted', 'du_return_on_equity',
                           'inc_revenue_rate', 'inc_net_profit_rate']
        # C1/C2: 额外财务字段
        if self.params.min_ocf_ratio is not None:
            pershare_fields.append('s_fa_ocfps')
        extra_tables = {}
        if self.params.max_debt_ratio is not None:
            extra_tables['Balance'] = ['total_assets', 'total_liability']

        pershare_data = self.get_financial_fields_batch(stock_list, 'Pershareindex', pershare_fields)

        balance_data = {}
        if extra_tables:
            for table, fields in extra_tables.items():
                balance_data = self.get_financial_fields_batch(stock_list, table, fields)

        result = []
        no_eps = 0
        no_roe = 0
        no_revenue = 0
        no_profit = 0
        no_ocf = 0
        no_debt = 0

        for stock in stock_list:
            stock_data = pershare_data.get(stock, {})
            b_data = balance_data.get(stock, {})

            # PE = 价格 / 稀释EPS
            eps = stock_data.get('s_fa_eps_diluted')
            price = self.get_unadjusted_price(stock)
            if eps is None or eps <= 0 or price is None or price <= 0:
                no_eps += 1
                continue
            pe = price / eps

            if not (self.params.min_pe < pe <= self.params.max_pe):
                continue

            # ROE
            roe = stock_data.get('du_return_on_equity')
            if roe is None or roe <= self.params.min_roe:
                no_roe += 1
                continue

            # 营收增速
            revenue_growth = stock_data.get('inc_revenue_rate')
            if revenue_growth is None or revenue_growth <= self.params.min_revenue_growth:
                no_revenue += 1
                continue

            # 净利润增速
            profit_growth = stock_data.get('inc_net_profit_rate')
            if profit_growth is None or profit_growth <= self.params.min_profit_growth:
                no_profit += 1
                continue

            # PEG = PE / 净利润增速
            peg = pe / profit_growth
            if not (self.params.min_peg <= peg <= self.params.max_peg):
                continue

            # C1: 经营现金流/净利润比率
            if self.params.min_ocf_ratio is not None:
                ocfps = stock_data.get('s_fa_ocfps')
                if ocfps is None or eps is None or eps <= 0:
                    no_ocf += 1
                    continue
                ocf_ratio = ocfps / eps
                if ocf_ratio < self.params.min_ocf_ratio:
                    no_ocf += 1
                    continue

            # C2: 资产负债率
            if self.params.max_debt_ratio is not None:
                total_assets = b_data.get('total_assets')
                total_liability = b_data.get('total_liability')
                if total_assets and total_assets > 0 and total_liability is not None:
                    debt_ratio = total_liability / total_assets * 100
                    # 银行股（行业代码J）豁免
                    industry = self.get_industry(stock) or ''
                    if '银行' not in industry and debt_ratio > self.params.max_debt_ratio:
                        no_debt += 1
                        continue

            result.append(stock)

        detail_parts = [f'总{len(stock_list)}只, 无EPS={no_eps}, '
                       f'ROE不足={no_roe}, 营收增速不足={no_revenue}, '
                       f'净利润增速不足={no_profit}']
        if self.params.min_ocf_ratio is not None:
            detail_parts.append(f'OCF不足={no_ocf}')
        if self.params.max_debt_ratio is not None:
            detail_parts.append(f'负债率过高={no_debt}')
        detail_parts.append(f'通过={len(result)}')
        self.log(f'基本面详情: {", ".join(detail_parts)}')
        return result

    def _filter_tradable(self, stock_list: List[str]) -> List[str]:
        """去停牌、去涨停（持仓中的涨停不过滤）"""
        result = []
        holdings = set(self._current_holdings.keys())
        for stock in stock_list:
            if self.is_suspended(stock):
                continue
            if stock not in holdings and self.is_limit_up(stock):
                continue
            result.append(stock)
        return result

    def _calc_pe(self, stock: str) -> Optional[float]:
        """计算PE = 价格 / 稀释EPS"""
        eps = self.get_financial_field(stock, 'Pershareindex', 's_fa_eps_diluted')
        price = self.get_unadjusted_price(stock)
        if eps and eps > 0 and price and price > 0:
            return price / eps
        return None

    # ================================================================
    # 第二轮优化方法
    # ================================================================

    def _check_sell_signals(self):
        """E1/E2: 基本面驱动的主动卖出信号"""
        if self.params.sell_max_pe is None and self.params.sell_min_dividend_yield is None:
            return

        for symbol in list(self._current_holdings.keys()):
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size <= 0:
                continue

            # E1: PE过高卖出
            if self.params.sell_max_pe is not None:
                pe = self._calc_pe(symbol)
                if pe is not None and pe > self.params.sell_max_pe:
                    sellable = self.get_sellable_volume(symbol)
                    if sellable > 0:
                        price = self.get_current_price(symbol)
                        if price and price > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                            self.sell(symbol, price, sellable)
                            del self._current_holdings[symbol]
                            self.log(f'PE过高({pe:.1f}>{self.params.sell_max_pe})，卖出: {symbol}')

            # E2: 股息率过低卖出
            if self.params.sell_min_dividend_yield is not None and symbol in self._current_holdings:
                dy = self.get_dividend_yield(symbol, use_avg=True)
                if dy is not None and dy < self.params.sell_min_dividend_yield:
                    sellable = self.get_sellable_volume(symbol)
                    if sellable > 0:
                        price = self.get_current_price(symbol)
                        if price and price > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                            self.sell(symbol, price, sellable)
                            del self._current_holdings[symbol]
                            self.log(f'股息率过低({dy:.2%}<{self.params.sell_min_dividend_yield:.2%})，卖出: {symbol}')

    def _sort_by_peg(self, stock_list: List[str]) -> List[str]:
        """B1: 按PEG升序排序"""
        peg_scores = {}
        for stock in stock_list:
            pe = self._calc_pe(stock)
            profit_growth = self.get_financial_field(stock, 'Pershareindex', 'inc_net_profit_rate')
            if pe is not None and profit_growth and profit_growth > 0:
                peg_scores[stock] = pe / profit_growth

        # 有PEG的排前面，无PEG的保持原序
        with_peg = [(s, peg_scores[s]) for s in stock_list if s in peg_scores]
        without_peg = [s for s in stock_list if s not in peg_scores]
        with_peg.sort(key=lambda x: x[1])
        return [s for s, _ in with_peg] + without_peg

    def _sort_by_composite_score(self, stock_list: List[str]) -> List[str]:
        """G1: 综合评分排序

        对已通过基本面筛选的股票，按多因子加权评分排序：
        - 股息率(25%): 越高越好
        - PE倒数(15%): 越低越好
        - PEG倒数(15%): 越低越好
        - ROE(20%): 越高越好
        - 营收增速(10%): 越高越好
        - 净利润增速(15%): 越高越好
        使用排名百分位归一化，避免极端值影响。
        """
        if not stock_list:
            return stock_list

        # 批量获取财务数据
        fields = ['s_fa_eps_diluted', 'du_return_on_equity', 'inc_revenue_rate', 'inc_net_profit_rate']
        fin_data = self.get_financial_fields_batch(stock_list, 'Pershareindex', fields)

        # 收集各指标原始值
        raw = {}
        for stock in stock_list:
            m = {}
            dy = self.get_dividend_yield(stock, use_avg=True)
            if dy is not None and dy > 0:
                m['dy'] = dy

            pe = self._calc_pe(stock)
            if pe is not None and pe > 0:
                m['pe_inv'] = 1.0 / pe

            sd = fin_data.get(stock, {})
            profit_growth = sd.get('inc_net_profit_rate')
            if pe is not None and profit_growth is not None and profit_growth > 0:
                m['peg_inv'] = profit_growth / pe  # = 1/PEG

            roe = sd.get('du_return_on_equity')
            if roe is not None:
                m['roe'] = roe

            rev_growth = sd.get('inc_revenue_rate')
            if rev_growth is not None:
                m['rev_growth'] = rev_growth

            if profit_growth is not None:
                m['profit_growth'] = profit_growth

            raw[stock] = m

        # 排名归一化 + 加权求和
        weights = {
            'dy': 0.25, 'pe_inv': 0.15, 'peg_inv': 0.15,
            'roe': 0.20, 'rev_growth': 0.10, 'profit_growth': 0.15
        }
        scores = {stock: 0.0 for stock in stock_list}

        for metric_name, weight in weights.items():
            valid = [(stock, raw[stock].get(metric_name)) for stock in stock_list
                     if metric_name in raw[stock]]
            if not valid:
                continue
            valid.sort(key=lambda x: x[1])
            n = len(valid)
            for rank, (stock, _) in enumerate(valid):
                scores[stock] += weight * (rank + 1) / n

        sorted_stocks = sorted(stock_list, key=lambda s: scores[s], reverse=True)
        return sorted_stocks

    def _filter_by_industry_diversity(self, stock_list: List[str]) -> List[str]:
        """行业分散: 每行业最多持仓N只

        按股息率排序后，每个行业只保留排名最高的N只股票。
        持仓中的股票有豁免权，不受此限制影响。
        """
        if not stock_list or self.params.max_industry_stocks is None:
            return stock_list

        max_per_industry = self.params.max_industry_stocks
        holdings = set(self._current_holdings.keys())

        # 统计各行业的股票数量（持仓中已有的股票不计入）
        industry_counts = {}
        result = []

        for stock in stock_list:
            # 持仓中的股票直接保留
            if stock in holdings:
                result.append(stock)
                continue

            industry = self.get_industry(stock) or '未知'
            if industry not in industry_counts:
                industry_counts[industry] = 0

            if industry_counts[industry] < max_per_industry:
                result.append(stock)
                industry_counts[industry] += 1

        industries_used = len(industry_counts)
        self.log(f'行业分散(每行业最多{max_per_industry}只): {len(stock_list)} -> {len(result)} 只, 覆盖{industries_used}个行业')
        return result

    def on_backtest_end(self):
        pass

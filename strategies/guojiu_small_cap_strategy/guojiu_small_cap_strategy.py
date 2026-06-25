import datetime as dt_module
from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('guojiu_small_cap',
                   default_kwargs={'max_stocks': 8},
                   backtest_config={'cash': 100000,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中小综指'})
class GuojiuSmallCapStrategy(StockSelectionStrategy):
    """国九小市值策略 - 基于国九条筛选的小市值选股

    克隆自聚宽文章: https://www.joinquant.com/view/community/detail/47791
    原始策略: 国九小市值策略【年化100.5%|回撤25.6%】

    选股逻辑：
    1. 从中小企业板指数(399101.SZ)成分股中筛选，过滤ST/退市/科创北交/创业板/次新股/停牌/涨跌停
    2. 国九条过滤：净利润>0、归母净利润>0、营业收入>1亿
    3. 按总市值升序排列，取前N只
    4. 过滤股价超过上限的股票

    调仓规则：
    - 每周调仓（原版run_weekly(weekly_adjustment, 2)即每2周，但涨停/止损/补仓每周执行）
    - 动态持股数量：根据指数均线位置调整（3~6只）
    - 1月和4月空仓
    - 持仓中昨日涨停的股票，若今日打开涨停则卖出
    - 个股止损9%，盈利100%止盈
    - 市场大跌止损（指数成分股平均跌幅>=5%时清仓）
    - 涨停卖出后次日补仓

    已知偏差（vs聚宽原版）：
    - 原版空仓月份持有银华日利ETF，本版空仓月份直接清仓持有现金
    - 原版审计意见过滤默认关闭，本版同样不实现
    - 原版涨停检查使用1分钟数据，本版使用日线数据近似判断
    - 原版市值使用valuation.market_cap（聚宽内置），本版用cap_stk*price（总股本×股价）
    """

    params = (
        ('rebalance_freq', 'weekly'),
        ('max_stocks', 8),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        # 策略特有参数
        ('min_market_cap', 10),          # 最小市值（亿）
        ('max_market_cap', 1e8),         # 最大市值（亿），实际不限制
        ('max_stock_price', 50),         # 股价上限
        ('min_ipo_days', 375),           # 上市天数下限
        ('skip_months', (1, 4)),         # 空仓月份
        ('stoploss_limit', 0.09),        # 个股止损线
        ('take_profit_ratio', 2.0),      # 止盈线（成本价的倍数）
        ('market_stoploss', 0.05),       # 市场大跌止损参数
        ('ma_period', 10),              # 动态持股MA周期
        # MA差值与持股数量映射阈值
        ('ma_diff_tiers', (500, 200, -200, -500)),
        ('ma_stock_nums', (5, 6, 8, 9, 10)),
        # 优化参数（默认禁用，启用时才生效）
        ('max_volatility', None),        # 日波动率上限（如0.05），过滤近20日波动率超限的股票
        ('volatility_lookback', 20),     # 波动率计算回溯天数
        ('max_same_industry', None),     # 同行业最多持股数（如2）
        ('max_debt_ratio', None),        # 资产负债率上限（如0.60）
        ('min_roe', None),               # ROE下限（如0.08）
        ('keep_existing', False),        # 已持有且仍在候选池的股票优先保留
        ('min_revenue_growth', None),    # 营收同比增长率下限（如0.0表示正增长）
        ('switch_threshold', None),      # 换仓阈值：新候选股市值需比当前持仓小此比例才换仓（如0.05表示5%）
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._high_limit_list: List[str] = []
        self._lifecycle_manager = None
        self._reason_to_sell = ''       # 记录卖出原因: 'limitup' or 'stoploss'
        self._limitup_sold_stocks: List[str] = []  # 记录因涨停打开而卖出的股票
        self._target_list: List[str] = []  # 当前目标持仓列表

    def _get_lifecycle_manager(self):
        if self._lifecycle_manager is None:
            try:
                from core.stock_lifecycle import get_lifecycle_manager
                self._lifecycle_manager = get_lifecycle_manager()
            except Exception:
                pass
        return self._lifecycle_manager

    def on_bar(self, bar):
        # 每日检查涨停和止损
        self._prepare_high_limit_list()
        self._check_limit_up()
        self._check_stoploss()
        self._check_market_stoploss()
        # 涨停卖出后次日补仓
        self._check_remain_amount()
        super().on_bar(bar)

    def select_stocks(self) -> List[str]:
        current_date = self.get_current_date()
        if current_date is None:
            return []

        # 检查是否为空仓月份
        month = current_date.month if isinstance(current_date, (dt_module.date, dt_module.datetime)) else None
        if month and month in self.params.skip_months:
            self.log(f'空仓月份({month}月)，清仓')
            return []

        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        # 步骤1: 基本过滤
        filtered = self._filter_basic(pool)
        self.log(f'基本过滤: {len(pool)} -> {len(filtered)} 只')

        if not filtered:
            self.log('基本过滤后无股票')
            return []

        # 步骤2: 国九条过滤（净利润>0、归母净利润>0、营收>1亿）+ 市值过滤
        filtered = self._filter_guojio(filtered)
        self.log(f'国九条+市值过滤: -> {len(filtered)} 只')

        if not filtered:
            self.log('国九条过滤后无股票')
            return []

        # 步骤2b: 优化过滤（默认禁用，启用时才生效）
        if self.params.max_volatility is not None:
            filtered = self._filter_volatility(filtered)
            self.log(f'波动率过滤(<{self.params.max_volatility}): -> {len(filtered)} 只')
            if not filtered:
                return []

        if self.params.max_debt_ratio is not None:
            filtered = self._filter_debt_ratio(filtered)
            self.log(f'负债率过滤(<{self.params.max_debt_ratio}): -> {len(filtered)} 只')
            if not filtered:
                return []

        if self.params.min_roe is not None:
            filtered = self._filter_roe(filtered)
            self.log(f'ROE过滤(>{self.params.min_roe}): -> {len(filtered)} 只')
            if not filtered:
                return []

        if self.params.min_revenue_growth is not None:
            filtered = self._filter_revenue_growth(filtered)
            self.log(f'营收增长过滤(>{self.params.min_revenue_growth}): -> {len(filtered)} 只')
            if not filtered:
                return []

        # 步骤3: 按市值升序排列
        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []

        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])

        # 步骤4: 动态持股数量
        dynamic_num = self._adjust_stock_num()
        if dynamic_num == 0:
            self.log('MA指示指数大跌，空仓')
            return []

        max_stocks = min(dynamic_num, self.params.max_stocks)
        max_stocks = int(max_stocks)
        self.log(f'动态持股数量: {max_stocks} (MA调整={dynamic_num}, 上限={self.params.max_stocks})')

        # 步骤5: 取前N只，过滤股价超限
        # 与原版一致：取 stock_num*3 只候选，再过滤股价
        candidates = sorted_stocks[:max_stocks * 3]

        # 优化：换仓阈值 - 减少不必要换仓
        if self.params.switch_threshold is not None and self.params.switch_threshold > 0:
            selected = self._apply_switch_threshold(candidates, max_stocks, market_caps)
        else:
            selected = []
            # 优化：keep_existing - 已持有且仍在候选池的优先保留
            if self.params.keep_existing:
                holdings = set(self._current_holdings.keys())
                for stock, mc in candidates:
                    if stock in holdings and len(selected) < max_stocks:
                        price = self.get_current_price(stock)
                        if price and price > 0:
                            selected.append(stock)
                # 剩余位置从候选中按市值升序补充
                for stock, mc in candidates:
                    if len(selected) >= max_stocks:
                        break
                    if stock in selected:
                        continue
                    price = self.get_current_price(stock)
                    # 已持有的股票不受股价上限限制
                    if stock not in self._current_holdings and price and price > self.params.max_stock_price:
                        continue
                    selected.append(stock)
            else:
                for stock, mc in candidates:
                    if len(selected) >= max_stocks:
                        break
                    price = self.get_current_price(stock)
                    # 已持有的股票不受股价上限限制
                    if stock not in self._current_holdings and price and price > self.params.max_stock_price:
                        continue
                    selected.append(stock)

        # 优化：行业分散 - 限制同行业最多持股数
        if self.params.max_same_industry is not None:
            selected = self._apply_industry_limit(selected, sorted_stocks, max_stocks, market_caps)

        # 保存目标列表
        self._target_list = selected[:]

        self.log(f'选股结果: {len(pool)} -> {len(selected)} 只')
        for stock in selected:
            mc = market_caps.get(stock, 0)
            price = self.get_current_price(stock)
            price_str = f'{price:.2f}' if price else 'N/A'
            self.log(f'  {stock} | 市值: {mc / 1e8:.2f}亿 | 价格: {price_str}')

        return selected

    # ================================================================
    # 过滤方法
    # ================================================================

    def _filter_basic(self, stock_list: List[str]) -> List[str]:
        """基本过滤：科创北交、ST/退市、次新股、停牌、涨跌停"""
        result = []
        holdings = set(self._current_holdings.keys())

        for stock in stock_list:
            # 过滤科创板(68)、北交所(8开头/4开头)
            # 与原版一致：过滤30/68/8/4开头的股票
            code = stock.split('.')[0] if '.' in stock else stock
            if code.startswith('30') or code.startswith('68') or code.startswith('8') or code.startswith('4'):
                continue

            # 过滤ST和退市
            lifecycle = self._get_lifecycle_manager()
            if lifecycle is not None:
                info = lifecycle._data.get(stock)
                if info and info.get('name'):
                    name = info['name']
                    if 'ST' in name or '*' in name or '退' in name:
                        continue

            # 过滤次新股
            if lifecycle is not None:
                list_date_str = lifecycle.get_list_date(stock)
                if list_date_str:
                    try:
                        list_date = dt_module.date.fromisoformat(list_date_str[:10])
                        current_date = self.get_current_date()
                        if current_date:
                            if isinstance(current_date, dt_module.datetime):
                                current = current_date.date()
                            else:
                                current = current_date
                            if (current - list_date).days < self.params.min_ipo_days:
                                continue
                    except (ValueError, TypeError):
                        pass

            # 过滤停牌
            if self.is_suspended(stock):
                continue

            # 过滤涨停（持仓中的不过滤，与原版一致）
            if stock not in holdings and self.is_limit_up(stock):
                continue

            # 过滤跌停（持仓中的不过滤，与原版一致）
            if stock not in holdings and self.is_limit_down(stock):
                continue

            result.append(stock)

        return result

    def _filter_guojio(self, stock_list: List[str]) -> List[str]:
        """国九条过滤 + 市值过滤：净利润>0、归母净利润>0、营业收入>1亿、市值范围"""
        if not stock_list:
            return []

        # 批量获取财务数据（QMT Income 表字段名）
        # revenue = 营业收入, net_profit_incl_min_int_inc_after = 净利润, minority_int_inc = 少数股东损益
        income_fields = ['net_profit_incl_min_int_inc_after', 'minority_int_inc', 'revenue']
        income_data = self.get_financial_fields_batch(stock_list, 'Income', income_fields)

        result = []
        for stock in stock_list:
            i_data = income_data.get(stock, {})

            # 净利润 > 0 (net_profit_incl_min_int_inc_after)
            net_profit = i_data.get('net_profit_incl_min_int_inc_after')
            if net_profit is None or net_profit <= 0:
                continue

            # 归属于母公司所有者的净利润 > 0 = 净利润 - 少数股东损益
            minority = i_data.get('minority_int_inc') or 0
            np_parent = net_profit - minority
            if np_parent <= 0:
                continue

            # 营业收入 > 1亿
            revenue = i_data.get('revenue')
            if revenue is None or revenue <= 1e8:
                continue

            # 市值过滤（使用流通股本×股价，与聚宽valuation.market_cap一致）
            market_cap = self._get_market_cap(stock)
            if market_cap is not None:
                cap_yi = market_cap / 1e8
                if cap_yi < self.params.min_market_cap:
                    continue
                if cap_yi > self.params.max_market_cap:
                    continue

            result.append(stock)

        return result

    # ================================================================
    # 优化过滤方法（默认禁用，启用时才生效）
    # ================================================================

    def _filter_volatility(self, stock_list: List[str]) -> List[str]:
        """波动率过滤：剔除近N日日收益率标准差超过阈值的股票"""
        if not stock_list:
            return []
        lookback = self.params.volatility_lookback
        threshold = self.params.max_volatility
        result = []
        for stock in stock_list:
            closes = self.get_close_prices(stock, lookback + 1)
            if closes is None or len(closes) < lookback + 1:
                # 数据不足，保留（避免过滤过多）
                result.append(stock)
                continue
            # 计算日收益率
            returns = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes)) if closes[i - 1] > 0]
            if len(returns) < lookback:
                result.append(stock)
                continue
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            std = variance ** 0.5
            if std <= threshold:
                result.append(stock)
        return result

    def _filter_debt_ratio(self, stock_list: List[str]) -> List[str]:
        """负债率过滤：剔除资产负债率超过阈值的股票
        资产负债率 = 总负债 / 总资产
        Balance表: total_liabilities(负债合计), total_assets(资产总计)
        """
        if not stock_list:
            return []
        fields = ['total_liabilities', 'total_assets']
        data = self.get_financial_fields_batch(stock_list, 'Balance', fields)
        result = []
        for stock in stock_list:
            b_data = data.get(stock, {})
            total_liab = b_data.get('total_liabilities')
            total_assets = b_data.get('total_assets')
            if total_liab is None or total_assets is None or total_assets <= 0:
                # 数据缺失，保留
                result.append(stock)
                continue
            debt_ratio = total_liab / total_assets
            if debt_ratio <= self.params.max_debt_ratio:
                result.append(stock)
        return result

    def _filter_roe(self, stock_list: List[str]) -> List[str]:
        """ROE过滤：剔除ROE低于阈值的股票
        Pershareindex表: s_fa_roe(净资产收益率加权)
        """
        if not stock_list:
            return []
        fields = ['s_fa_roe']
        data = self.get_financial_fields_batch(stock_list, 'Pershareindex', fields)
        result = []
        for stock in stock_list:
            p_data = data.get(stock, {})
            roe = p_data.get('s_fa_roe')
            if roe is None:
                # 数据缺失，保留
                result.append(stock)
                continue
            # s_fa_roe 通常以百分比表示（如15.0表示15%），需要转换
            roe_value = roe / 100.0 if abs(roe) > 1 else roe
            if roe_value >= self.params.min_roe:
                result.append(stock)
        return result

    def _filter_revenue_growth(self, stock_list: List[str]) -> List[str]:
        """营收增长过滤：剔除营收同比增长率低于阈值的股票
        Income表: revenue(本期营收), yst_oper_income(去年同期营收)
        若字段不存在则尝试用 growth_rate 系列
        """
        if not stock_list:
            return []
        fields = ['revenue', 'yst_oper_income']
        data = self.get_financial_fields_batch(stock_list, 'Income', fields)
        result = []
        for stock in stock_list:
            i_data = data.get(stock, {})
            revenue = i_data.get('revenue')
            prev_revenue = i_data.get('yst_oper_income')
            if revenue is None or prev_revenue is None or prev_revenue <= 0:
                # 数据缺失，保留
                result.append(stock)
                continue
            growth = (revenue - prev_revenue) / prev_revenue
            if growth >= self.params.min_revenue_growth:
                result.append(stock)
        return result

    def _apply_industry_limit(self, selected: List[str], sorted_stocks: List,
                               max_stocks: int, market_caps: Dict[str, float]) -> List[str]:
        """行业分散：限制同行业最多持股数，超限的用后续候选补充"""
        limit = self.params.max_same_industry
        if limit is None or limit <= 0:
            return selected

        # 统计已选股票的行业分布
        industry_count: Dict[str, int] = {}
        final_selected = []
        rejected = []
        for stock in selected:
            industry = self.get_industry(stock) or '未知'
            if industry_count.get(industry, 0) < limit:
                final_selected.append(stock)
                industry_count[industry] = industry_count.get(industry, 0) + 1
            else:
                rejected.append(stock)

        # 从后续候选中补充被剔除的位置
        if rejected:
            selected_set = set(final_selected)
            for stock, mc in sorted_stocks:
                if len(final_selected) >= max_stocks:
                    break
                if stock in selected_set:
                    continue
                industry = self.get_industry(stock) or '未知'
                if industry_count.get(industry, 0) < limit:
                    price = self.get_current_price(stock)
                    if stock not in self._current_holdings and price and price > self.params.max_stock_price:
                        continue
                    final_selected.append(stock)
                    selected_set.add(stock)
                    industry_count[industry] = industry_count.get(industry, 0) + 1

        return final_selected

    def _apply_switch_threshold(self, candidates, max_stocks: int, market_caps: Dict[str, float]) -> List[str]:
        """换仓阈值：新候选股市值需比当前持仓小threshold比例才值得换仓

        逻辑：
        1. 以市值升序排列的候选池为基础
        2. 当前持仓中仍在候选池的，优先保留
        3. 新候选股只有当市值比被替换的持仓小超过threshold比例时才换入
        4. 这样减少不必要换仓，降低交易成本
        """
        threshold = self.params.switch_threshold
        holdings = set(self._current_holdings.keys())

        # 构建有效候选列表（过滤股价超限）
        valid = []
        for stock, mc in candidates:
            price = self.get_current_price(stock)
            # 已持有的股票不受股价上限限制
            if stock not in holdings and price and price > self.params.max_stock_price:
                continue
            valid.append(stock)

        if not valid:
            return []

        # 理想组合：市值最小的前N只
        ideal = valid[:max_stocks]

        # 无持仓时直接返回理想组合
        if not holdings:
            return ideal

        # 构建最终组合
        result = []
        result_set = set()

        # 步骤1：当前持仓中在理想组合内的，直接保留
        for stock in ideal:
            if stock in holdings:
                result.append(stock)
                result_set.add(stock)

        # 步骤2：对剩余位置，在新候选和未入选持仓间选择
        for stock in ideal:
            if len(result) >= max_stocks:
                break
            if stock in result_set:
                continue

            # 查找未入选的持仓中市值最小的
            unselected_holdings = [h for h in holdings if h not in result_set and h in market_caps]
            if unselected_holdings:
                best_holding = min(unselected_holdings, key=lambda h: market_caps[h])
                best_holding_mc = market_caps[best_holding]
                candidate_mc = market_caps.get(stock, float('inf'))

                # 只有当新候选市值比持仓小超过threshold时才换仓
                if candidate_mc < best_holding_mc * (1 - threshold):
                    result.append(stock)
                    result_set.add(stock)
                else:
                    # 保留当前持仓，不换仓
                    result.append(best_holding)
                    result_set.add(best_holding)
            else:
                result.append(stock)
                result_set.add(stock)

        # 步骤3：如果位置还没填满，从候选池补充
        for stock in valid:
            if len(result) >= max_stocks:
                break
            if stock not in result_set:
                result.append(stock)
                result_set.add(stock)

        return result[:max_stocks]

    def _get_market_cap(self, stock: str) -> Optional[float]:
        """获取总市值 = 总股本 × 当前价格

        与聚宽 valuation.market_cap 一致，使用财报中的总股本(cap_stk)
        注意：不能使用 xtdata.get_instrument_detail 的 FloatVolume/TotalVolume，
        因为那些是当前值而非历史值，会导致历史回测时市值计算严重偏高
        """
        price = self.get_unadjusted_price(stock)
        if price is None or price <= 0:
            return None

        # 优先使用财报中的总股本(cap_stk)，这是point-in-time数据
        cap_stk = self.get_financial_field(stock, 'Balance', 'cap_stk')
        if cap_stk and cap_stk > 0:
            return cap_stk * price

        # 回退：使用 total_equity/bps 估算总股本
        total_equity = self.get_financial_field(stock, 'Balance', 'total_equity')
        bps = self.get_financial_field(stock, 'Pershareindex', 's_fa_bps')
        if total_equity and total_equity > 0 and bps and bps > 0:
            total_shares = total_equity / bps
            return total_shares * price
        return None

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        """计算总市值 = 总股本 × 当前股价（与聚宽 valuation.market_cap 一致）"""
        result = {}
        for stock in stocks:
            mc = self._get_market_cap(stock)
            if mc and mc > 0:
                result[stock] = mc
        return result

    # ================================================================
    # 动态持股数量
    # ================================================================

    def _adjust_stock_num(self) -> int:
        """根据指数MA与收盘价差值动态调整持股数量

        与原版一致：使用399101.XSHE（中小企业板指数）
        差值 >= 500: 3只 (市场强势，集中持仓)
        200 <= 差值 < 500: 3只
        -200 <= 差值 < 200: 4只 (震荡)
        -500 <= 差值 < -200: 5只
        差值 < -500: 6只 (市场弱势，分散持仓)
        """
        # 使用中小企业板指数代码
        index_symbol = '399101.SZ'
        closes = self.get_close_prices(index_symbol, self.params.ma_period + 1)

        if closes is None or len(closes) < self.params.ma_period:
            self.log(f'MA数据不足，使用默认持股数 {self.params.max_stocks}')
            return self.params.max_stocks

        ma = sum(closes[-self.params.ma_period:]) / self.params.ma_period
        current_close = closes[-1]
        diff = current_close - ma

        tiers = self.params.ma_diff_tiers
        nums = self.params.ma_stock_nums

        if diff >= tiers[0]:
            result = nums[0]
        elif diff >= tiers[1]:
            result = nums[1]
        elif diff >= tiers[2]:
            result = nums[2]
        elif diff >= tiers[3]:
            result = nums[3]
        else:
            result = nums[4]

        self.log(f'动态持股: 收盘={current_close:.2f}, MA{self.params.ma_period}={ma:.2f}, 差值={diff:.2f}, 持股={result}')
        return result

    # ================================================================
    # 涨停检查
    # ================================================================

    def _prepare_high_limit_list(self):
        """检查持仓中昨日涨停的股票（与原版一致：使用日线close vs high_limit）"""
        holdings = self.get_current_holdings()
        self._high_limit_list = []

        for symbol in holdings:
            ohlcv = self.get_ohlcv_data(symbol, 2)
            if ohlcv and len(ohlcv) >= 2:
                prev = ohlcv[-2]
                prev_close = prev.get('close', 0)
                prev_high_limit = prev.get('high_limit', 0)
                # 昨日收盘价等于涨停价则为涨停
                if prev_high_limit > 0 and prev_close >= prev_high_limit - 0.005:
                    self._high_limit_list.append(symbol)

    def _check_limit_up(self):
        """昨日涨停今天开板的卖出（与原版一致）"""
        if not self._high_limit_list:
            return

        for symbol in self._high_limit_list[:]:
            # 如果今天不再涨停，则卖出
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
                            self._reason_to_sell = 'limitup'
                            self._limitup_sold_stocks.append(symbol)
            else:
                self.log(f'涨停继续持有: {symbol}')

    # ================================================================
    # 涨停卖出后次日补仓（与原版 check_remain_amount 一致）
    # ================================================================

    def _check_remain_amount(self):
        """涨停卖出后次日补仓

        原版逻辑：如果卖出原因是涨停打开，则在次日买入新股票补位
        如果卖出原因是止损，则不补仓
        """
        if self._reason_to_sell != 'limitup':
            return

        self._reason_to_sell = ''
        current_holdings = list(self._current_holdings.keys())

        if len(current_holdings) < self.params.max_stocks and self._target_list:
            # 计算需要买入的股票数量
            num_to_buy = min(len(self._limitup_sold_stocks), self.params.max_stocks - len(current_holdings))
            # 从目标列表中选择不在已卖出列表中的股票
            candidates = [s for s in self._target_list if s not in self._limitup_sold_stocks][:num_to_buy]

            if candidates:
                cash = self.get_cash()
                per_stock_value = cash / len(candidates) if candidates else 0
                for stock in candidates:
                    price = self.get_current_price(stock)
                    if price and price > 0 and not self.is_suspended(stock) and not self.is_limit_up(stock):
                        volume = int(per_stock_value / price / 100) * 100
                        if volume > 0:
                            self.buy(stock, price, volume)
                            self._current_holdings[stock] = self._current_holdings.get(stock, 0) + volume
                            self.log(f'涨停补仓，买入: {stock} {volume}股 @ {price:.2f}')

        self._limitup_sold_stocks = []

    # ================================================================
    # 止损止盈
    # ================================================================

    def _check_stoploss(self):
        """个股止损止盈检查（与原版一致：止损后设置 reason_to_sell='stoploss'）"""
        for symbol in list(self._current_holdings.keys()):
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size <= 0:
                continue

            price = self.get_current_price(symbol)
            if not price or price <= 0:
                continue

            # 获取持仓成本
            avg_cost = self._get_avg_cost(symbol)
            if avg_cost is None or avg_cost <= 0:
                continue

            # 盈利100%止盈
            if price >= avg_cost * self.params.take_profit_ratio:
                sellable = self.get_sellable_volume(symbol)
                if sellable > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                    self.sell(symbol, price, sellable)
                    if symbol in self._current_holdings:
                        del self._current_holdings[symbol]
                    self.log(f'收益{self.params.take_profit_ratio - 1:.0%}止盈，卖出: {symbol}')

            # 止损
            elif price < avg_cost * (1 - self.params.stoploss_limit):
                sellable = self.get_sellable_volume(symbol)
                if sellable > 0 and not self.is_limit_down(symbol) and not self.is_suspended(symbol):
                    self.sell(symbol, price, sellable)
                    if symbol in self._current_holdings:
                        del self._current_holdings[symbol]
                    self.log(f'止损(亏损{self.params.stoploss_limit:.0%})，卖出: {symbol}')
                    self._reason_to_sell = 'stoploss'

    def _check_market_stoploss(self):
        """市场大跌止损（与原版一致：计算中小企业板指数所有成分股的平均跌幅）

        原版逻辑：
        stock_df = get_price(security=get_index_stocks('399101.XSHE'), ...)
        down_ratio = abs((stock_df['close'] / stock_df['open'] - 1).mean())
        if down_ratio >= g.stoploss_market: 清仓
        """
        if self.params.market_stoploss is None or self.params.market_stoploss <= 0:
            return

        if not self._current_holdings:
            return

        # 获取中小企业板指数成分股当日涨跌情况
        pool = self.get_stock_pool()
        if not pool:
            return

        # 批量获取成分股当日OHLCV
        total_ratio = 0
        count = 0
        for stock in pool:
            ohlcv = self.get_ohlcv_data(stock, 1)
            if ohlcv and len(ohlcv) >= 1:
                open_price = ohlcv[-1].get('open', 0)
                close_price = ohlcv[-1].get('close', 0)
                if open_price > 0:
                    total_ratio += close_price / open_price - 1
                    count += 1

        if count > 0:
            avg_ratio = abs(total_ratio / count)
            if avg_ratio >= self.params.market_stoploss:
                self.log(f'市场大跌止损: 成分股平均跌幅{avg_ratio:.2%} >= {self.params.market_stoploss:.2%}，清仓')
                self._sell_all()
                self._reason_to_sell = 'stoploss'

    def _get_avg_cost(self, symbol: str) -> Optional[float]:
        """获取持仓均价"""
        try:
            if self.executor and hasattr(self.executor, 'get_avg_cost'):
                cost = self.executor.get_avg_cost(symbol)
                if cost and cost > 0:
                    return cost
        except Exception:
            pass
        return None

    def on_backtest_end(self):
        pass

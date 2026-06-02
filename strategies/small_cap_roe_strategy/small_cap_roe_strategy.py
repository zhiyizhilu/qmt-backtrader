import datetime as dt_module
from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from core.data_adapter import get_limit_ratio
from strategies import register_strategy


@register_strategy('small_cap_roe',
                   default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0003,
                                    'start_date': '2019-01-01', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指'})
class SmallCapRoeStrategy(StockSelectionStrategy):
    """小市值ROE/ROA策略 - 月度调仓选股

    克隆自聚宽文章: https://www.joinquant.com/post/45510
    原始策略: ROE>15%且ROA>10%的小市值股票，月度调仓

    选股逻辑：
    1. 过滤科创板、北交所、ST、停牌、次新、涨跌停、高价股(>10元)
    2. 基本面筛选：ROE > 15%, ROA > 10%
    3. 按市值升序排列（小市值优先）
    4. 剔除黑名单（30天内持有过且30天内涨停过的股票）

    调仓规则：
    - 月度调仓，等权重持仓
    - 最多持仓10只股票
    - 每日检查持仓中昨日涨停的股票，若打开涨停则卖出
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('min_roe', 0.15),
        ('min_roa', 0.10),
        ('max_price', 10.0),
        ('limit_days', 30),
        ('min_ipo_days', 250),
        ('max_volatility', None),
        ('stop_loss_pct', None),
        ('max_same_industry', None),
        ('min_market_cap', None),
        ('keep_existing', None),
        ('momentum_period', None),
        ('max_debt_ratio', None),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._history_hold_list: List[List[str]] = []
        self._not_buy_again_list: List[str] = []
        self._just_sold: List[str] = []
        self._high_limit_list: List[str] = []
        self._lifecycle_manager = None
        self._purchase_prices: Dict[str, float] = {}

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
        prev_holdings = set(self._current_holdings.keys())
        super().on_bar(bar)
        curr_holdings = set(self._current_holdings.keys())
        for symbol in curr_holdings - prev_holdings:
            price = self.get_unadjusted_price(symbol)
            if price and price > 0:
                self._purchase_prices[symbol] = price
        for symbol in prev_holdings - curr_holdings:
            if symbol in self._purchase_prices:
                del self._purchase_prices[symbol]

    def _prepare_high_limit_list(self):
        holdings = self.get_current_holdings()
        self._high_limit_list = []

        for symbol in holdings:
            ohlcv = self.get_ohlcv_data(symbol, 2)
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

        hold_list = list(holdings.keys())
        self._history_hold_list.append(hold_list)
        if len(self._history_hold_list) >= self.params.limit_days:
            self._history_hold_list = self._history_hold_list[-self.params.limit_days:]

        temp_set = set()
        for hl in self._history_hold_list:
            for stock in hl:
                temp_set.add(stock)
        self._not_buy_again_list = list(temp_set)

    def _check_limit_up(self):
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
                            self._just_sold.append(symbol)
                            if len(self._just_sold) >= self.params.limit_days:
                                self._just_sold = self._just_sold[-self.params.max_stocks:]
                            self.log(f'涨停打开，卖出: {symbol}')

        position_count = len(self._current_holdings)
        if self.params.max_stocks > position_count and position_count != 0:
            self._refill_positions()

    def _refill_positions(self):
        target_stocks = self.select_stocks()
        if not target_stocks:
            return

        position_count = len(self._current_holdings)
        if self.params.max_stocks <= position_count:
            return

        cash = self.get_cash()
        slots = self.params.max_stocks - position_count
        psize = cash / slots if slots > 0 else 0

        from core.data_adapter import get_trade_unit, validate_trade_volume

        for s in target_stocks:
            if s in self._current_holdings:
                continue
            price = self.get_current_price(s)
            if not price or price <= 0:
                continue
            if self.is_suspended(s) or self.is_limit_up(s):
                continue
            buy_volume = int(psize / price / get_trade_unit(s)) * get_trade_unit(s)
            is_valid, _ = validate_trade_volume(s, buy_volume)
            if is_valid and buy_volume > 0:
                self.buy(s, price, buy_volume)
                self._current_holdings[s] = self._current_holdings.get(s, 0) + buy_volume
                self.log(f'补仓买入: {s}, 数量: {buy_volume}, 价格: {price:.2f}')
                if len(self._current_holdings) >= self.params.max_stocks:
                    break

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        filtered = self._filter_kcbj_stock(pool)
        self.log(f'过滤科创北交: {len(pool)} -> {len(filtered)} 只')

        filtered = self._filter_st_stock(filtered)
        self.log(f'过滤ST: -> {len(filtered)} 只')

        filtered = self._filter_suspended(filtered)
        self.log(f'过滤停牌: -> {len(filtered)} 只')

        filtered = self._filter_new_stock(filtered)
        self.log(f'过滤次新股: -> {len(filtered)} 只')

        filtered = self._filter_limit_stocks(filtered)
        self.log(f'过滤涨跌停: -> {len(filtered)} 只')

        filtered = self._filter_high_price(filtered)
        self.log(f'过滤高价股(>{self.params.max_price}元): -> {len(filtered)} 只')

        filtered = self._filter_fundamental(filtered)
        self.log(f'基本面筛选(ROE>{self.params.min_roe}, ROA>{self.params.min_roa}): -> {len(filtered)} 只')

        filtered = self._filter_debt_ratio(filtered)
        if self.params.max_debt_ratio is not None:
            self.log(f'负债率过滤(>{self.params.max_debt_ratio}): -> {len(filtered)} 只')

        filtered = self._filter_volatility(filtered)
        if self.params.max_volatility is not None:
            self.log(f'波动率过滤(>{self.params.max_volatility}): -> {len(filtered)} 只')

        filtered = self._filter_momentum(filtered)
        if self.params.momentum_period is not None:
            self.log(f'动量过滤(周期{self.params.momentum_period}): -> {len(filtered)} 只')

        filtered = self._filter_stop_loss(filtered)
        if self.params.stop_loss_pct is not None:
            self.log(f'止损过滤(>{self.params.stop_loss_pct*100:.0f}%): -> {len(filtered)} 只')

        if not filtered:
            self.log('筛选后无股票')
            return []

        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []

        market_caps = self._filter_min_market_cap(market_caps)
        if self.params.min_market_cap is not None:
            self.log(f'市值下限过滤(>={self.params.min_market_cap/1e8:.0f}亿): -> {len(market_caps)} 只')

        if not market_caps:
            self.log('市值过滤后无股票')
            return []

        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])

        recent_limit_up_list = self._get_recent_limit_up_stocks(
            [s for s, _ in sorted_stocks], self.params.limit_days
        )
        black_list = list(
            set(self._not_buy_again_list).intersection(set(recent_limit_up_list))
        )
        target_list = [stock for stock, _ in sorted_stocks if stock not in black_list]
        self.log(f'黑名单过滤: {len(sorted_stocks)} -> {len(target_list)} 只 (黑名单{len(black_list)}只)')

        target_list = self._filter_industry_concentration(target_list)
        if self.params.max_same_industry is not None:
            self.log(f'行业分散(同行业<={self.params.max_same_industry}只): -> {len(target_list)} 只')

        max_stocks = self.params.max_stocks
        selected = target_list[:max_stocks]

        selected = self._apply_keep_existing(selected)
        if self.params.keep_existing is not None:
            self.log(f'换手率控制: 保留现有持仓 -> {len(selected)} 只')

        self.log(f'选股结果: {len(pool)} -> {len(selected)} 只')
        for stock, mc in sorted_stocks[:max_stocks]:
            if stock in selected:
                self.log(f'  {stock} | 市值: {mc / 1e8:.2f}亿')

        return selected

    def _filter_kcbj_stock(self, stock_list: List[str]) -> List[str]:
        result = []
        for stock in stock_list:
            code = stock.split('.')[0] if '.' in stock else stock
            if code.startswith('4') or code.startswith('8') or code.startswith('68'):
                continue
            result.append(stock)
        return result

    def _filter_st_stock(self, stock_list: List[str]) -> List[str]:
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

    def _filter_suspended(self, stock_list: List[str]) -> List[str]:
        return [s for s in stock_list if not self.is_suspended(s)]

    def _filter_new_stock(self, stock_list: List[str]) -> List[str]:
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

    def _filter_limit_stocks(self, stock_list: List[str]) -> List[str]:
        result = []
        holdings = set(self._current_holdings.keys())
        for stock in stock_list:
            if stock in holdings:
                result.append(stock)
                continue
            if self.is_limit_up(stock):
                continue
            if self.is_limit_down(stock):
                continue
            result.append(stock)
        return result

    def _filter_high_price(self, stock_list: List[str]) -> List[str]:
        result = []
        holdings = set(self._current_holdings.keys())
        for stock in stock_list:
            if stock in holdings:
                result.append(stock)
                continue
            price = self.get_unadjusted_price(stock)
            if price is not None and price >= self.params.max_price:
                continue
            result.append(stock)
        return result

    def _filter_fundamental(self, stock_list: List[str]) -> List[str]:
        if not stock_list:
            return []

        pershare_fields = ['du_return_on_equity']
        pershare_data = self.get_financial_fields_batch(stock_list, 'Pershareindex', pershare_fields)

        balance_fields = ['total_assets', 'total_equity']
        balance_data = self.get_financial_fields_batch(stock_list, 'Balance', balance_fields)

        result = []
        no_roe = 0
        no_balance = 0
        low_roa = 0
        for stock in stock_list:
            stock_pershare = pershare_data.get(stock, {})
            roe = stock_pershare.get('du_return_on_equity')
            if roe is None or roe <= self.params.min_roe:
                no_roe += 1
                continue

            stock_balance = balance_data.get(stock, {})
            total_assets = stock_balance.get('total_assets')
            total_equity = stock_balance.get('total_equity')

            if total_assets and total_assets > 0 and total_equity and total_equity > 0:
                roa = roe * (total_equity / total_assets)
                if roa <= self.params.min_roa:
                    low_roa += 1
                    continue
            else:
                no_balance += 1
                continue

            result.append(stock)

        self.log(f'基本面筛选详情: 总{len(stock_list)}只, 无ROE={no_roe}, 无资产负债数据={no_balance}, ROA不足={low_roa}, 通过={len(result)}')
        return result

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        no_price = 0
        no_cap_data = 0

        for stock in stocks:
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                no_price += 1
                continue

            total_equity = self.get_financial_field(stock, 'Balance', 'total_equity')
            bps = self.get_financial_field(stock, 'Pershareindex', 's_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                market_cap = total_shares * price
                result[stock] = market_cap
            elif total_equity and total_equity > 0:
                result[stock] = total_equity
            else:
                no_cap_data += 1

        if no_price > 0 or no_cap_data > 0:
            self.log(f'市值计算: {len(stocks)} 只 -> {len(result)} 只有有效数据 '
                     f'(无价格={no_price}, 无股本数据={no_cap_data})')

        return result

    def _get_recent_limit_up_stocks(self, stock_list: List[str], recent_days: int) -> List[str]:
        result = []
        for stock in stock_list:
            ohlcv = self.get_ohlcv_data(stock, recent_days + 1)
            if not ohlcv or len(ohlcv) < 2:
                continue

            has_limit_up = False
            for i in range(1, len(ohlcv)):
                prev_close = ohlcv[i - 1].get('close', 0)
                curr_close = ohlcv[i].get('close', 0)
                if prev_close > 0 and curr_close > 0:
                    limit_ratio = get_limit_ratio(stock)
                    limit_price = round(prev_close * (1 + limit_ratio), 2)
                    if curr_close >= limit_price - 0.005:
                        has_limit_up = True
                        break

            if has_limit_up:
                result.append(stock)
        return result

    def _filter_volatility(self, stock_list: List[str]) -> List[str]:
        if self.params.max_volatility is None:
            return stock_list
        result = []
        for stock in stock_list:
            closes = self.get_close_prices(stock, 20)
            if len(closes) < 5:
                result.append(stock)
                continue
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
            if returns:
                avg_ret = sum(returns) / len(returns)
                variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
                daily_vol = variance ** 0.5
                if daily_vol > self.params.max_volatility:
                    continue
            result.append(stock)
        return result

    def _filter_stop_loss(self, stock_list: List[str]) -> List[str]:
        if self.params.stop_loss_pct is None:
            return stock_list
        result = []
        for stock in stock_list:
            if stock in self._purchase_prices:
                purchase_price = self._purchase_prices[stock]
                current_price = self.get_unadjusted_price(stock)
                if current_price and purchase_price > 0:
                    loss_pct = (current_price - purchase_price) / purchase_price
                    if loss_pct <= -self.params.stop_loss_pct:
                        self.log(f'止损过滤: {stock}, 亏损{loss_pct*100:.1f}%')
                        continue
            result.append(stock)
        return result

    def _filter_industry_concentration(self, stock_list: List[str]) -> List[str]:
        if self.params.max_same_industry is None:
            return stock_list
        industry_count: Dict[str, int] = {}
        result = []
        for stock in stock_list:
            industry = self.get_industry(stock)
            if industry is None:
                result.append(stock)
                continue
            count = industry_count.get(industry, 0)
            if count >= self.params.max_same_industry:
                continue
            industry_count[industry] = count + 1
            result.append(stock)
        return result

    def _filter_min_market_cap(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        if self.params.min_market_cap is None:
            return market_caps
        result = {}
        for stock, cap in market_caps.items():
            if cap >= self.params.min_market_cap:
                result[stock] = cap
            else:
                self.log(f'市值过小过滤: {stock}, 市值{cap/1e8:.2f}亿 < {self.params.min_market_cap/1e8:.2f}亿')
        return result

    def _filter_momentum(self, stock_list: List[str]) -> List[str]:
        if self.params.momentum_period is None:
            return stock_list
        result = []
        for stock in stock_list:
            closes = self.get_close_prices(stock, self.params.momentum_period + 1)
            if len(closes) >= 2:
                momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
                if momentum <= 0:
                    continue
            result.append(stock)
        return result

    def _filter_debt_ratio(self, stock_list: List[str]) -> List[str]:
        if self.params.max_debt_ratio is None:
            return stock_list
        result = []
        for stock in stock_list:
            try:
                total_assets = self.get_financial_field(stock, 'Balance', 'total_assets')
                total_liab = self.get_financial_field(stock, 'Balance', 'total_liabilities')
                if total_assets and total_assets > 0 and total_liab is not None:
                    debt_ratio = total_liab / total_assets
                    if debt_ratio > self.params.max_debt_ratio:
                        continue
            except Exception:
                pass
            result.append(stock)
        return result

    def _apply_keep_existing(self, selected: List[str]) -> List[str]:
        if self.params.keep_existing is None:
            return selected
        holdings = set(self._current_holdings.keys())
        kept = [s for s in selected if s in holdings]
        new = [s for s in selected if s not in holdings]
        kept_set = set(kept)
        for stock in holdings:
            if stock not in kept_set and stock in selected:
                pass
        for stock in holdings:
            if stock not in kept_set:
                kept.append(stock)
        return kept + new

    def _get_portfolio_value(self) -> float:
        cash = self.get_cash()
        holdings_value = 0
        for symbol, volume in self._current_holdings.items():
            price = self.get_current_price(symbol)
            if price and price > 0:
                holdings_value += price * volume
        return cash + holdings_value

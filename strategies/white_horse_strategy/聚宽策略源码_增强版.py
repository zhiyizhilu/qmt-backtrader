# 克隆自聚宽文章：https://www.joinquant.com/post/50043
# 增强版：增加详细日志输出，用于与QMT策略对比分析

# 导入函数库
from jqdata import *

# 初始化函数，设定基准等等
def initialize(context):
	# 设定沪深300作为基准
	set_benchmark('000300.XSHG')
	# 开启动态复权模式(真实价格)
	set_option('use_real_price', True)
	#防止未来函数
	set_option("avoid_future_data", True)
	# 输出内容到日志 log.info()
	log.info('初始函数开始运行且全局只运行一次')
	# 过滤掉order系列API产生的比error级别低的log
	log.set_level('order', 'error')

	# 股票池
	g.buy_stock_count = 5
	g.check_out_lists = []
	g.market_temperature = "warm"
	
	### 股票相关设定 ###
	set_order_cost(OrderCost(close_tax=0.001, open_commission=0.00012, close_commission=0.00012, min_commission=5),
				   type='stock')

	# 开盘前运行
	run_monthly(before_market_open, 1, time='5:00', reference_security='000300.XSHG')
	run_monthly(my_trade, 1, time='9:45', reference_security='000300.XSHG')
	# 收盘后运行
	run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


## 开盘时运行函数
def my_trade(context):
	# 买卖
	adjust_position(context, g.check_out_lists)

## 收盘后运行函数
def after_market_close(context):
	log.info('#########################################################################################\n\n')


# 自定义下单
def order_target_value_(security, value):
	if value == 0:
		log.debug("卖出 %s" % (get_name(security)))
	else:
		log.debug("买入 %s ，市值： %f" % (get_name(security), value))
	return order_target_value(security, value)


# 开仓
def open_position(security, value):
	order = order_target_value_(security, value)
	if order != None and order.filled > 0:
		return True
	return False


# 平仓
def close_position(position):
	security = position.security
	order = order_target_value_(security, 0)
	if order != None:
		if order.status == OrderStatus.held and order.filled == order.amount:
			return True
	return False


# 交易
def adjust_position(context, buy_stocks):
	for stock in context.portfolio.positions:
		current_data = get_current_data()
		nosell_1 = context.portfolio.positions[stock].price >= current_data[stock].high_limit
		sell_2 = stock not in buy_stocks
		if sell_2 and not nosell_1:
			log.info("调出平仓：[%s]" % (stock))
			position = context.portfolio.positions[stock]
			close_position(position)
		else:
			log.info("已持仓，本次不买入：[%s]" % (stock))

	position_count = len(context.portfolio.positions)
	if g.buy_stock_count > position_count:
		value = context.portfolio.cash / (g.buy_stock_count - position_count)
		for stock in buy_stocks[:g.buy_stock_count]:
			if stock not in context.portfolio.positions:
				if open_position(stock, value):
					if len(context.portfolio.positions) >= g.buy_stock_count:
						break

def get_name(stk):
    return get_security_info(stk).display_name+':'+stk[:6]


def Market_temperature(context):
    index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']
    market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))
    
    # ====== 增强日志：输出市场温度计算的中间值 ======
    log.info("【市场温度】index300近5日均值=%.2f, 220日最低=%.2f, 220日最高=%.2f, market_height=%.4f" % 
             (mean(index300[-5:]), min(index300), max(index300), market_height))
    log.info("【市场温度】近60日最高=%.2f, 全期最低=%.2f, 60日/全期最低比率=%.4f" %
             (max(index300[-60:]), min(index300), max(index300[-60:]) / min(index300)))
    
    if market_height < 0.20:
        g.market_temperature = "cold"
    elif market_height > 0.90:
        g.market_temperature = "hot"
    elif max(index300[-60:]) / min(index300) > 1.20:
        g.market_temperature = "warm"
    
    log.info("【市场温度】最终判断=%s" % g.market_temperature)


## 开盘前运行函数
def before_market_open(context):
    Market_temperature(context)
    g.check_out_lists = []
    current_data = get_current_data()
    check_date = context.previous_date - datetime.timedelta(days=200)
    all_stocks = list(get_all_securities(date=check_date).index)
    all_stocks = get_index_stocks("000300.XSHG")
    
    # 过滤
    all_stocks = [stock for stock in all_stocks if not (
            (current_data[stock].day_open == current_data[stock].high_limit) or
            (current_data[stock].day_open == current_data[stock].low_limit) or
            current_data[stock].paused or
            current_data[stock].is_st or
            ('ST' in current_data[stock].name) or
            ('*' in current_data[stock].name) or
            ('退' in current_data[stock].name) or
            (stock.startswith('30')) or
            (stock.startswith('68')) or
            (stock.startswith('8')) or
            (stock.startswith('4'))
    )]
    
    # ====== 增强日志：输出过滤后的股票池 ======
    log.info("【股票池】沪深300总数=300, 过滤后=%d只" % len(all_stocks))
    
    # ====== 增强日志：批量获取所有候选股票的详细指标 ======
    # 获取所有过滤后股票的PB、ROA、现金流等指标
    q_all = query(
        valuation.code,
        valuation.pb_ratio,
        valuation.market_cap,
        indicator.roa,
        indicator.inc_return,
        indicator.inc_net_profit_year_on_year,
        indicator.adjusted_profit,
        cash_flow.subtotal_operate_cash_inflow,
        cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit,
    ).filter(
        valuation.code.in_(all_stocks)
    )
    df_all = get_fundamentals(q_all)
    df_all.columns = ['code', 'pb_ratio', 'market_cap', 'roa', 'inc_return',
                       'inc_net_profit_yoy', 'adjusted_profit', 
                       'cash_inflow', 'cash_quality']
    
    # ====== 增强日志：输出指标数据统计 ======
    valid_pb = df_all[df_all['pb_ratio'] > 0].shape[0]
    valid_roa = df_all[df_all['roa'].notna()].shape[0]
    valid_cash = df_all[(df_all['cash_inflow'] > 0) & (df_all['adjusted_profit'] > 0)].shape[0]
    log.info("【指标统计】有PB>0=%d, 有ROA=%d, 有现金流>0且adjusted_profit>0=%d" % (valid_pb, valid_roa, valid_cash))
    
    # ====== 增强日志：按市场温度输出筛选过程中的每一步 ======
    if g.market_temperature == "cold":
        # 冷市: PB<1, cash_inflow>0, adjusted_profit>0, cash_quality>2.0, inc_return>1.5, profit_yoy>-15
        log.info("【冷市筛选】条件: PB>0且<1, cash_inflow>0, adjusted_profit>0, cash_quality>2.0, inc_return>1.5, profit_yoy>-15")
        
        # 分步筛选日志
        step1 = df_all[(df_all['pb_ratio'] > 0) & (df_all['pb_ratio'] < 1)]
        log.info("  步骤1(PB>0且<1): %d只" % step1.shape[0])
        
        step2 = step1[(step1['cash_inflow'] > 0) & (step1['adjusted_profit'] > 0)]
        log.info("  步骤2(cash_inflow>0且adjusted_profit>0): %d只" % step2.shape[0])
        
        step3 = step2[step2['cash_quality'] > 2.0]
        log.info("  步骤3(cash_quality>2.0): %d只" % step3.shape[0])
        
        step4 = step3[step3['inc_return'] > 1.5]
        log.info("  步骤4(inc_return>1.5): %d只" % step4.shape[0])
        
        step5 = step4[step4['inc_net_profit_yoy'] > -15]
        log.info("  步骤5(profit_yoy>-15): %d只" % step5.shape[0])
        
        # 输出最终候选的详细指标
        if step5.shape[0] > 0:
            step5_sorted = step5.copy()
            step5_sorted['score'] = step5_sorted['roa'] / step5_sorted['pb_ratio']
            step5_sorted = step5_sorted.sort_values('score', ascending=False)
            log.info("【冷市候选】最终候选=%d只, 按ROA/PB排序:" % step5_sorted.shape[0])
            for _, row in step5_sorted.head(10).iterrows():
                log.info("  %s: PB=%.2f, ROA=%.2f%%, inc_return=%.2f%%, profit_yoy=%.2f%%, "
                         "cash_inflow=%.0f, adjusted_profit=%.0f, cash_quality=%.2f, score=%.2f" %
                         (row['code'][:6], row['pb_ratio'], row['roa']*100 if row['roa'] < 1 else row['roa'],
                          row['inc_return'], row['inc_net_profit_yoy'],
                          row['cash_inflow'], row['adjusted_profit'],
                          row['cash_quality'], row['score']))
        
        q = query(
            valuation.code, 
            ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow/indicator.adjusted_profit>2.0,
            indicator.inc_return > 1.5,
            indicator.inc_net_profit_year_on_year > -15,
        	valuation.code.in_(all_stocks)
        	).order_by(
        	(indicator.roa/valuation.pb_ratio).desc()
        ).limit(
        	g.buy_stock_count + 1
        )

    elif g.market_temperature == "warm":
        # 温市: PB<1, cash_inflow>0, adjusted_profit>0, cash_quality>1.0, inc_return>2.0, profit_yoy>0
        log.info("【温市筛选】条件: PB>0且<1, cash_inflow>0, adjusted_profit>0, cash_quality>1.0, inc_return>2.0, profit_yoy>0")
        
        step1 = df_all[(df_all['pb_ratio'] > 0) & (df_all['pb_ratio'] < 1)]
        log.info("  步骤1(PB>0且<1): %d只" % step1.shape[0])
        
        step2 = step1[(step1['cash_inflow'] > 0) & (step1['adjusted_profit'] > 0)]
        log.info("  步骤2(cash_inflow>0且adjusted_profit>0): %d只" % step2.shape[0])
        
        step3 = step2[step2['cash_quality'] > 1.0]
        log.info("  步骤3(cash_quality>1.0): %d只" % step3.shape[0])
        
        step4 = step3[step3['inc_return'] > 2.0]
        log.info("  步骤4(inc_return>2.0): %d只" % step4.shape[0])
        
        step5 = step4[step4['inc_net_profit_yoy'] > 0]
        log.info("  步骤5(profit_yoy>0): %d只" % step5.shape[0])
        
        if step5.shape[0] > 0:
            step5_sorted = step5.copy()
            step5_sorted['score'] = step5_sorted['roa'] / step5_sorted['pb_ratio']
            step5_sorted = step5_sorted.sort_values('score', ascending=False)
            log.info("【温市候选】最终候选=%d只, 按ROA/PB排序:" % step5_sorted.shape[0])
            for _, row in step5_sorted.head(10).iterrows():
                log.info("  %s: PB=%.2f, ROA=%.2f%%, inc_return=%.2f%%, profit_yoy=%.2f%%, "
                         "cash_inflow=%.0f, adjusted_profit=%.0f, cash_quality=%.2f, score=%.2f" %
                         (row['code'][:6], row['pb_ratio'], row['roa']*100 if row['roa'] < 1 else row['roa'],
                          row['inc_return'], row['inc_net_profit_yoy'],
                          row['cash_inflow'], row['adjusted_profit'],
                          row['cash_quality'], row['score']))
        
        q = query(
            valuation.code, 
            ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow/indicator.adjusted_profit>1.0,
            indicator.inc_return > 2.0,
            indicator.inc_net_profit_year_on_year > 0,
        	valuation.code.in_(all_stocks)
        	).order_by(
        	(indicator.roa/valuation.pb_ratio).desc()
        ).limit(
        	g.buy_stock_count + 1
        )

    elif g.market_temperature == "hot":
        # 热市: PB>3, cash_inflow>0, adjusted_profit>0, cash_quality>0.5, inc_return>3.0, profit_yoy>20
        log.info("【热市筛选】条件: PB>3, cash_inflow>0, adjusted_profit>0, cash_quality>0.5, inc_return>3.0, profit_yoy>20")
        
        step1 = df_all[df_all['pb_ratio'] > 3]
        log.info("  步骤1(PB>3): %d只" % step1.shape[0])
        
        step2 = step1[(step1['cash_inflow'] > 0) & (step1['adjusted_profit'] > 0)]
        log.info("  步骤2(cash_inflow>0且adjusted_profit>0): %d只" % step2.shape[0])
        
        step3 = step2[step2['cash_quality'] > 0.5]
        log.info("  步骤3(cash_quality>0.5): %d只" % step3.shape[0])
        
        step4 = step3[step3['inc_return'] > 3.0]
        log.info("  步骤4(inc_return>3.0): %d只" % step4.shape[0])
        
        step5 = step4[step4['inc_net_profit_yoy'] > 20]
        log.info("  步骤5(profit_yoy>20): %d只" % step5.shape[0])
        
        if step5.shape[0] > 0:
            step5_sorted = step5.copy()
            step5_sorted = step5_sorted['roa']  # 热市按ROA排序
            step5_sorted = step5.copy()
            step5_sorted['score'] = step5_sorted['roa']
            step5_sorted = step5_sorted.sort_values('score', ascending=False)
            log.info("【热市候选】最终候选=%d只, 按ROA排序:" % step5_sorted.shape[0])
            for _, row in step5_sorted.head(10).iterrows():
                log.info("  %s: PB=%.2f, ROA=%.2f%%, inc_return=%.2f%%, profit_yoy=%.2f%%, "
                         "cash_inflow=%.0f, adjusted_profit=%.0f, cash_quality=%.2f, score=%.2f" %
                         (row['code'][:6], row['pb_ratio'], row['roa']*100 if row['roa'] < 1 else row['roa'],
                          row['inc_return'], row['inc_net_profit_yoy'],
                          row['cash_inflow'], row['adjusted_profit'],
                          row['cash_quality'], row['score']))
        
        q = query(
            valuation.code, 
            ).filter(
            valuation.pb_ratio > 3,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow/indicator.adjusted_profit>0.5,
            indicator.inc_return > 3.0,
            indicator.inc_net_profit_year_on_year > 20,
        	valuation.code.in_(all_stocks)
        	).order_by(
        	indicator.roa.desc()
        ).limit(
        	g.buy_stock_count + 1
        )
    
    check_out_lists = list(get_fundamentals(q).code)
    g.check_out_lists = check_out_lists
    
    # ====== 增强日志：输出最终选股结果 ======
    log.info("【最终选股】今日股票池=%s" % g.check_out_lists)
    
    # ====== 增强日志：输出选中股票的详细指标 ======
    if len(check_out_lists) > 0:
        q_selected = query(
            valuation.code,
            valuation.pb_ratio,
            indicator.roa,
            indicator.inc_return,
            indicator.inc_net_profit_year_on_year,
            indicator.adjusted_profit,
            cash_flow.subtotal_operate_cash_inflow,
        ).filter(
            valuation.code.in_(check_out_lists)
        )
        df_selected = get_fundamentals(q_selected)
        log.info("【选中股票指标】")
        for _, row in df_selected.iterrows():
            stock_name = get_name(row['code'])
            log.info("  %s: PB=%.4f, ROA=%.4f, inc_return=%.4f, profit_yoy=%.4f, "
                     "adjusted_profit=%.0f, cash_inflow=%.0f" %
                     (stock_name, row['pb_ratio'], row['roa'], row['inc_return'],
                      row['inc_net_profit_year_on_year'], row['adjusted_profit'],
                      row['subtotal_operate_cash_inflow']))

"""精确计算四大行买卖价差导致的价格侵蚀比例

计算方法：
1. 加载tick数据，提取每笔tick的ask1/bid1/lastPrice
2. 计算买入侵蚀 = (ask1 - lastPrice) / lastPrice
3. 计算卖出侵蚀 = (lastPrice - bid1) / lastPrice
4. 计算双向侵蚀 = (ask1 - bid1) / lastPrice
5. 分别计算每只银行股的统计值（均值、中位数、分位数）
6. 计算整体加权平均侵蚀比例，作为回测框架的slippage参数

重要发现（2026-05验证）：
- 51%的tick ask1=lastPrice（买入无侵蚀），48%的tick bid1=lastPrice（卖出无侵蚀）
- 仅1.1%的tick双边都有价差，0.3%的tick双边相等
- 因此应使用含零均值（0.067%）而非非零均值（0.14%）作为slippage
- 修正后slippage=0.0007，双向滑点成本0.14%（之前0.28%高估一倍）
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from xtquant import xtdata

xtdata.enable_hello = False

BANK_STOCKS = {
    '工商银行': '601398.SH',
    '农业银行': '601288.SH',
    '建设银行': '601939.SH',
    '中国银行': '601988.SH',
}


def main():
    symbols = list(BANK_STOCKS.values())
    start_date = '20260501'
    end_date = '20260528'

    # 下载数据
    for s in symbols:
        xtdata.download_history_data(s, 'tick', start_date, end_date)

    results = {}
    all_buy_erosions = []
    all_sell_erosions = []
    all_bidirectional = []

    for name, symbol in BANK_STOCKS.items():
        data = xtdata.get_market_data_ex(
            [], [symbol], period='tick',
            start_time=start_date, end_time=end_date,
            count=-1, dividend_type='none'
        )
        if symbol not in data or data[symbol].empty:
            print(f'{name}({symbol}): 无数据')
            continue

        df = data[symbol]
        df = df[df['lastPrice'] > 0].copy()

        # 提取ask1和bid1
        df['ask1'] = df['askPrice'].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 and x[0] > 0 else np.nan
        )
        df['bid1'] = df['bidPrice'].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 and x[0] > 0 else np.nan
        )

        valid = df.dropna(subset=['ask1', 'bid1'])
        if valid.empty:
            continue

        # 计算侵蚀比例
        buy_erosion = (valid['ask1'] - valid['lastPrice']) / valid['lastPrice']  # 买入侵蚀
        sell_erosion = (valid['lastPrice'] - valid['bid1']) / valid['lastPrice']  # 卖出侵蚀
        bidirectional = (valid['ask1'] - valid['bid1']) / valid['lastPrice']  # 双向价差

        # 过滤异常值（价差为0的情况，可能是涨跌停）
        buy_nonzero = buy_erosion[buy_erosion > 0]
        sell_nonzero = sell_erosion[sell_erosion > 0]
        bidi_nonzero = bidirectional[bidirectional > 0]

        results[name] = {
            'buy_erosion_mean': buy_erosion.mean(),
            'buy_erosion_median': buy_erosion.median(),
            'buy_erosion_p75': buy_erosion.quantile(0.75),
            'buy_erosion_p90': buy_erosion.quantile(0.90),
            'buy_erosion_nonzero_mean': buy_nonzero.mean() if len(buy_nonzero) > 0 else 0,
            'sell_erosion_mean': sell_erosion.mean(),
            'sell_erosion_median': sell_erosion.median(),
            'sell_erosion_nonzero_mean': sell_nonzero.mean() if len(sell_nonzero) > 0 else 0,
            'bidirectional_mean': bidirectional.mean(),
            'bidirectional_median': bidirectional.median(),
            'bidirectional_p75': bidirectional.quantile(0.75),
            'bidirectional_p90': bidirectional.quantile(0.90),
            'bidirectional_nonzero_mean': bidi_nonzero.mean() if len(bidi_nonzero) > 0 else 0,
            'zero_spread_pct': (bidirectional == 0).sum() / len(bidirectional) * 100,
            'sample_count': len(valid),
        }

        all_buy_erosions.extend(buy_erosion.values)
        all_sell_erosions.extend(sell_erosion.values)
        all_bidirectional.extend(bidirectional.values)

    # 打印结果
    print('=' * 90)
    print('四大行买卖价差价格侵蚀比例分析')
    print('=' * 90)
    print(f'数据区间: {start_date} ~ {end_date}')
    print()

    print(f'{"标的":<10} {"买入侵蚀(均值)":<16} {"买入侵蚀(中位)":<16} {"买入侵蚀(P75)":<16} {"买入侵蚀(P90)":<16} {"零价差占比":<12}')
    print('-' * 86)
    for name, r in results.items():
        print(f'{name:<8} {r["buy_erosion_mean"]*100:>10.4f}%    {r["buy_erosion_median"]*100:>10.4f}%    '
              f'{r["buy_erosion_p75"]*100:>10.4f}%    {r["buy_erosion_p90"]*100:>10.4f}%    {r["zero_spread_pct"]:>8.1f}%')

    print()
    print(f'{"标的":<10} {"卖出侵蚀(均值)":<16} {"卖出侵蚀(中位)":<16} {"双向价差(均值)":<16} {"双向价差(中位)":<16} {"非零双向(均值)":<16}')
    print('-' * 90)
    for name, r in results.items():
        print(f'{name:<8} {r["sell_erosion_mean"]*100:>10.4f}%    {r["sell_erosion_median"]*100:>10.4f}%    '
              f'{r["bidirectional_mean"]*100:>10.4f}%    {r["bidirectional_median"]*100:>10.4f}%    '
              f'{r["bidirectional_nonzero_mean"]*100:>10.4f}%')

    # 整体统计
    all_buy = np.array(all_buy_erosions)
    all_sell = np.array(all_sell_erosions)
    all_bidi = np.array(all_bidirectional)

    print()
    print('=' * 90)
    print('整体统计（四大行合并）')
    print('=' * 90)
    print(f'总样本数: {len(all_buy)}')
    print(f'买入侵蚀(均值): {all_buy.mean()*100:.4f}%')
    print(f'买入侵蚀(中位): {np.median(all_buy)*100:.4f}%')
    print(f'买入侵蚀(P75):  {np.percentile(all_buy, 75)*100:.4f}%')
    print(f'买入侵蚀(P90):  {np.percentile(all_buy, 90)*100:.4f}%')
    print(f'卖出侵蚀(均值): {all_sell.mean()*100:.4f}%')
    print(f'卖出侵蚀(中位): {np.median(all_sell)*100:.4f}%')
    print(f'双向价差(均值): {all_bidi.mean()*100:.4f}%')
    print(f'双向价差(中位): {np.median(all_bidi)*100:.4f}%')
    print(f'双向价差(P75):  {np.percentile(all_bidi, 75)*100:.4f}%')
    print(f'双向价差(P90):  {np.percentile(all_bidi, 90)*100:.4f}%')

    # 计算回测框架应使用的slippage值
    # 框架的slippage是单向的：买入时 price * (1 + slippage)，卖出时 price * (1 - slippage)
    # 所以slippage应该等于单方向的平均侵蚀比例
    # 买入侵蚀 = (ask1 - lastPrice) / lastPrice
    # 卖出侵蚀 = (lastPrice - bid1) / lastPrice
    # 两者基本对称，取均值即可

    slippage_mean = all_buy.mean()  # 单向侵蚀均值
    slippage_median = np.median(all_buy)  # 单向侵蚀中位数
    slippage_p75 = np.percentile(all_buy, 75)  # 单向侵蚀P75

    print()
    print('=' * 90)
    print('回测框架 slippage 参数推荐值')
    print('=' * 90)
    print(f'框架逻辑: 买入价 = price * (1 + slippage), 卖出价 = price * (1 - slippage)')
    print(f'slippage = 单向价格侵蚀比例 (买入侵蚀 = 卖出侵蚀 ≈ 双向价差/2)')
    print()
    print(f'推荐值（均值）:   slippage = {slippage_mean:.6f} ({slippage_mean*100:.4f}%)')
    print(f'推荐值（中位）:   slippage = {slippage_median:.6f} ({slippage_median*100:.4f}%)')
    print(f'推荐值（P75）:    slippage = {slippage_p75:.6f} ({slippage_p75*100:.4f}%)')
    print(f'推荐值（保守P90）: slippage = {np.percentile(all_buy, 90):.6f} ({np.percentile(all_buy, 90)*100:.4f}%)')
    print()
    print(f'换仓一次的总侵蚀 = 买入侵蚀 + 卖出侵蚀 = {slippage_mean*2*100:.4f}% (均值)')
    print(f'当前 switch_threshold=0.003 时，换仓利润空间=0.3%')
    print(f'价差侵蚀占比 = {slippage_mean*2/0.003*100:.1f}%')
    print()

    # 计算不同threshold下的可行性
    print('=' * 90)
    print('不同 switch_threshold 下的价差侵蚀占比')
    print('=' * 90)
    commission = 0.0002  # 单向佣金
    total_cost_per_switch = slippage_mean * 2 + commission * 2  # 双向滑点+双向佣金
    print(f'单次换仓总成本(双向滑点+双向佣金): {total_cost_per_switch*100:.4f}%')
    print()
    for threshold in [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010]:
        cost_ratio = total_cost_per_switch / threshold * 100
        net_profit = threshold - total_cost_per_switch
        feasible = '可行' if net_profit > 0 else '不可行'
        print(f'  threshold={threshold:.3f}: 成本占比={cost_ratio:.1f}%, 净利润空间={net_profit*100:.4f}%, {feasible}')

    # 输出JSON格式供优化脚本使用
    print()
    print('=' * 90)
    print('JSON输出（供优化脚本使用）')
    print('=' * 90)
    import json
    output = {
        'slippage_mean': float(slippage_mean),
        'slippage_median': float(slippage_median),
        'slippage_p75': float(slippage_p75),
        'slippage_p90': float(np.percentile(all_buy, 90)),
        'bidirectional_mean': float(all_bidi.mean()),
        'total_cost_per_switch': float(total_cost_per_switch),
        'per_stock': {name: {k: float(v) for k, v in r.items()} for name, r in results.items()},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

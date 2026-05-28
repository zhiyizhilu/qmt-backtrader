"""银行轮动策略优化脚本 - 含买卖价差滑点

基于tick数据精确计算的买卖价差侵蚀比例：
  - 单向价格侵蚀(非零均值): 0.14%
  - 双向价差均值: 0.139%
  - 单次换仓总成本(双向滑点+双向佣金): 0.174%

将 slippage=0.0014 固化到回测框架中，模拟实盘买卖价差对成交价的影响。
框架逻辑：买入价 = price * (1 + 0.0014), 卖出价 = price * (1 - 0.0014)

优化目标：在含滑点的真实成本下，找到最优的 switch_threshold 和其他参数。
"""
import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir

STRATEGY_NAME = 'bank_rotation'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# 基于tick数据精确计算的买卖价差滑点
# 验证结论：51%的tick ask1=lastPrice(买入无侵蚀)，48%的tick bid1=lastPrice(卖出无侵蚀)
# 仅1.1%的tick双边都有价差，因此应使用含零均值而非非零均值
# 含零均值: 买入侵蚀0.067%, 卖出侵蚀0.072%, 平均0.0696% -> 取0.0007
SLIPPAGE = 0.0007


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              start_date=TRAIN_START, end_date=TRAIN_END, slippage=SLIPPAGE):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1m'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config['slippage'] = slippage
    config.setdefault('benchmark', '000300.SH')

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **merged_kwargs)
    results = api.run()

    result = api.get_result()
    metrics = {}
    if result:
        sr = result.sharpe_ratio()
        dd = result.max_drawdown()
        acc = result.account
        metrics['initial_capital'] = acc.initial_capital
        metrics['final_value'] = acc.dynamic_rights
        metrics['total_return_pct'] = acc.rate * 100
        metrics['sharpe_ratio'] = sr
        metrics['max_drawdown_pct'] = dd * 100
        if result.df is not None and len(result.df) > 0:
            days = len(result.df)
            years = days / 252
            annual_ret = (1 + acc.rate) ** (1 / years) - 1 if years > 0 else 0
            if isinstance(annual_ret, complex):
                annual_ret = annual_ret.real
            metrics['annual_return_pct'] = float(annual_ret) * 100
            metrics['trading_days'] = days
        metrics['fee'] = getattr(result, 'total_fee', 0)
        metrics['turnover'] = getattr(result, 'turnover', 0)
        metrics['label'] = label
        metrics['extra_params'] = extra_params
        metrics['slippage'] = slippage
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def run_out_of_sample_test(extra_params, label, slippage=SLIPPAGE):
    """使用固定验证集进行样本外测试"""
    in_sample = run_backtest_with_params(
        extra_params=extra_params,
        label=f'{label}_is', start_date=TRAIN_START, end_date=TRAIN_END, slippage=slippage)

    out_sample = run_backtest_with_params(
        extra_params=extra_params,
        label=f'{label}_oos', start_date=VALID_START, end_date=VALID_END, slippage=slippage)

    baseline_is = run_backtest_with_params(
        extra_params=None,
        label='slip_baseline_is', start_date=TRAIN_START, end_date=TRAIN_END, slippage=slippage)

    baseline_oos = run_backtest_with_params(
        extra_params=None,
        label='slip_baseline_oos', start_date=VALID_START, end_date=VALID_END, slippage=slippage)

    is_improvement = (in_sample.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
    oos_improvement = (out_sample.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100

    return {
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'valid_period': f'{VALID_START} ~ {VALID_END}',
        'in_sample_sharpe': in_sample.get('sharpe_ratio', 0),
        'out_sample_sharpe': out_sample.get('sharpe_ratio', 0),
        'baseline_is_sharpe': baseline_is.get('sharpe_ratio', 0),
        'baseline_oos_sharpe': baseline_oos.get('sharpe_ratio', 0),
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': oos_improvement / is_improvement if is_improvement != 0 else 0,
    }


def main():
    print('=' * 70)
    print('银行轮动策略优化 - 含买卖价差滑点')
    print('=' * 70)
    print(f'slippage = {SLIPPAGE} ({SLIPPAGE*100:.2f}%)')
    print(f'单次换仓总成本(双向滑点+双向佣金) = {(SLIPPAGE*2 + 0.0002*2)*100:.4f}%')
    print(f'测试集: {TRAIN_START} ~ {TRAIN_END}')
    print(f'验证集: {VALID_START} ~ {VALID_END}')
    print()

    # ===== 阶段1: 基线回测（含滑点） =====
    print('[1] 基线回测（含滑点, switch_threshold=0.003）...')
    baseline = run_backtest_with_params(
        extra_params=None, label='slip_baseline')
    print(f'  基线: 夏普={baseline.get("sharpe_ratio", 0):.4f}, '
          f'收益={baseline.get("total_return_pct", 0):.2f}%, '
          f'回撤={baseline.get("max_drawdown_pct", 0):.2f}%')

    # 无滑点基线对比
    print('[1b] 无滑点基线对比...')
    baseline_noslip = run_backtest_with_params(
        extra_params=None, label='slip_baseline_noslip', slippage=0.0)
    print(f'  无滑点: 夏普={baseline_noslip.get("sharpe_ratio", 0):.4f}, '
          f'收益={baseline_noslip.get("total_return_pct", 0):.2f}%')
    slip_impact = baseline_noslip.get('total_return_pct', 0) - baseline.get('total_return_pct', 0)
    print(f'  滑点侵蚀收益: {slip_impact:.2f}%')

    # ===== 阶段2: switch_threshold 优化 =====
    print()
    print('[2] switch_threshold 参数扫描...')
    thresholds = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010]
    threshold_results = []
    for t in thresholds:
        r = run_backtest_with_params(
            extra_params={'switch_threshold': t},
            label=f'slip_st_{t:.3f}')
        threshold_results.append(r)
        print(f'  threshold={t:.3f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    # 找最优threshold
    best_st = max(threshold_results, key=lambda x: x.get('sharpe_ratio', 0))
    best_threshold = best_st.get('extra_params', {}).get('switch_threshold', 0.003)
    print(f'\n  最优 switch_threshold = {best_threshold:.3f} (夏普={best_st.get("sharpe_ratio", 0):.4f})')

    # ===== 阶段3: spread_threshold 优化 =====
    print()
    print('[3] spread_threshold 参数扫描（配合最优switch_threshold）...')
    spreads = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    spread_results = []
    for s in spreads:
        r = run_backtest_with_params(
            extra_params={'switch_threshold': best_threshold, 'spread_threshold': s},
            label=f'slip_st{best_threshold:.3f}_sp{s:.3f}')
        spread_results.append(r)
        print(f'  spread={s:.3f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    best_sp = max(spread_results, key=lambda x: x.get('sharpe_ratio', 0))
    best_spread = best_sp.get('extra_params', {}).get('spread_threshold', 0.005)
    print(f'\n  最优 spread_threshold = {best_spread:.3f} (夏普={best_sp.get("sharpe_ratio", 0):.4f})')

    # ===== 阶段4: 其他优化项 =====
    best_params = {'switch_threshold': best_threshold, 'spread_threshold': best_spread}

    print()
    print('[4] 其他优化项测试...')

    # 4a: 波动率过滤
    print('  4a: 波动率过滤...')
    for vol in [0.02, 0.03, 0.04]:
        r = run_backtest_with_params(
            extra_params={**best_params, 'max_volatility': vol},
            label=f'slip_vol{vol:.2f}')
        print(f'    vol={vol:.2f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4b: 开盘不交易
    print('  4b: 开盘不交易...')
    for start, end in [('09:30', '09:45'), ('09:30', '10:00')]:
        r = run_backtest_with_params(
            extra_params={**best_params, 'no_trade_start': start, 'no_trade_end': end},
            label=f'slip_notrade_{start.replace(":","")}_{end.replace(":","")}')
        print(f'    {start}-{end}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4c: 收盘不交易
    print('  4c: 收盘不交易...')
    for close_start in ['14:45', '14:50', '14:55']:
        r = run_backtest_with_params(
            extra_params={**best_params, 'no_trade_close_start': close_start},
            label=f'slip_noclose_{close_start.replace(":","")}')
        print(f'    {close_start}后不交易: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4d: 每日最大换仓次数
    print('  4d: 每日最大换仓次数...')
    for max_t in [2, 3, 4]:
        r = run_backtest_with_params(
            extra_params={**best_params, 'max_daily_trades': max_t},
            label=f'slip_maxtrades{max_t}')
        print(f'    max_trades={max_t}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4e: 自适应阈值
    print('  4e: 自适应阈值...')
    for adaptive in [0.5, 1.0]:
        r = run_backtest_with_params(
            extra_params={**best_params, 'adaptive_threshold': adaptive},
            label=f'slip_adaptive{adaptive:.1f}')
        print(f'    adaptive={adaptive:.1f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4f: 扩展标的
    print('  4f: 扩展标的（交通银行）...')
    r = run_backtest_with_params(
        extra_params={**best_params, 'extra_stocks': {'交通银行': '601328.SH'}},
        label='slip_extra_jt')
    print(f'    +交通银行: 夏普={r.get("sharpe_ratio", 0):.4f}, '
          f'收益={r.get("total_return_pct", 0):.2f}%')

    # 4g: 比率确认
    print('  4g: 比率确认K线数...')
    for confirm in [2, 3, 5]:
        r = run_backtest_with_params(
            extra_params={**best_params, 'confirm_bars': confirm},
            label=f'slip_confirm{confirm}')
        print(f'    confirm={confirm}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%')

    # ===== 阶段5: 样本外验证 =====
    print()
    print('[5] 样本外验证（最优参数）...')
    oos_result = run_out_of_sample_test(best_params, 'slip_best')
    print(f'  测试集夏普: {oos_result["in_sample_sharpe"]:.4f}')
    print(f'  验证集夏普: {oos_result["out_sample_sharpe"]:.4f}')
    print(f'  基线测试集夏普: {oos_result["baseline_is_sharpe"]:.4f}')
    print(f'  基线验证集夏普: {oos_result["baseline_oos_sharpe"]:.4f}')
    print(f'  样本内改进: {oos_result["is_improvement_pct"]:.1f}%')
    print(f'  样本外改进: {oos_result["oos_improvement_pct"]:.1f}%')
    print(f'  衰减比: {oos_result["decay_ratio"]:.2f}')

    # ===== 汇总 =====
    print()
    print('=' * 70)
    print('优化汇总')
    print('=' * 70)
    print(f'滑点参数: slippage = {SLIPPAGE} ({SLIPPAGE*100:.2f}%)')
    print(f'单次换仓总成本: {(SLIPPAGE*2 + 0.0002*2)*100:.4f}%')
    print(f'最优参数: switch_threshold={best_threshold}, spread_threshold={best_spread}')
    print(f'基线(0.003, 含滑点): 夏普={baseline.get("sharpe_ratio", 0):.4f}, 收益={baseline.get("total_return_pct", 0):.2f}%')
    print(f'优化后(含滑点):     夏普={best_sp.get("sharpe_ratio", 0):.4f}, 收益={best_sp.get("total_return_pct", 0):.2f}%')
    improvement = (best_sp.get('sharpe_ratio', 0) - baseline.get('sharpe_ratio', 0)) / abs(baseline.get('sharpe_ratio', 1)) * 100
    print(f'夏普提升: {improvement:.1f}%')


if __name__ == '__main__':
    main()

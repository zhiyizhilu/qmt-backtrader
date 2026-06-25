"""Round 2 优化脚本 - 探索新方向

10项新方向优化（均未在Round 1中测试）：
1. 波动率过滤(3%) - 比Round1的5%更低
2. 波动率过滤(4%) - 匹配small_cap策略最优阈值
3. 换仓阈值(5%) - 来自ETF轮动成功经验
4. 换仓阈值(10%) - 更高阈值减少换手
5. 止盈50%(1.5x) - 比当前100%(2.0x)更低
6. 市场止损(3%) - 比当前5%更敏感
7. MA周期20日 - 比当前10日更长
8. 持仓8只 - 比当前6只更分散
9. 最小市值5亿 - 比当前10亿更低
10. 空仓月份(1,2,4) - 增加2月空仓（春节效应）
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
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'guojiu_small_cap'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 固定的测试集/验证集时间边界
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'
POOL = '中小综指'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(pool, '000300.SH')
    config.setdefault('benchmark', benchmark)

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.load_financial_data(sector=pool)
    api.add_stock_selection_strategy(strategy_class, **merged_kwargs)
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
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# Round 2 的10项新方向优化配置
R2_OPTIMIZATIONS = [
    ('r2_baseline', '基线(当前参数)', None),
    ('r2_opt01_volatility_3pct', '波动率过滤(3%)', {'max_volatility': 0.03}),
    ('r2_opt02_volatility_4pct', '波动率过滤(4%)', {'max_volatility': 0.04}),
    ('r2_opt03_switch_5pct', '换仓阈值(5%)', {'switch_threshold': 0.05}),
    ('r2_opt04_switch_10pct', '换仓阈值(10%)', {'switch_threshold': 0.10}),
    ('r2_opt05_take_profit_15x', '止盈50%(1.5x)', {'take_profit_ratio': 1.5}),
    ('r2_opt06_market_stoploss_3pct', '市场止损(3%)', {'market_stoploss': 0.03}),
    ('r2_opt07_ma_period_20', 'MA周期20日', {'ma_period': 20}),
    ('r2_opt08_max_stocks_8', '持仓8只', {'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}),
    ('r2_opt09_min_market_cap_5', '最小市值5亿', {'min_market_cap': 5}),
    ('r2_opt10_skip_months_124', '空仓月份(1,2,4)', {'skip_months': (1, 2, 4)}),
]


def run_all_optimizations():
    """运行所有优化回测"""
    print("=" * 60)
    print("Round 2 优化回测 (测试集 2020-04-28 ~ 2024-04-28)")
    print("=" * 60)
    results = []
    for label, name, params in R2_OPTIMIZATIONS:
        print(f"\n>>> 运行: {name} ({label})")
        try:
            result = run_backtest_with_params(
                strategy_name=STRATEGY_NAME,
                extra_params=params,
                label=label)
            sharpe = result.get('sharpe_ratio', 0)
            total_ret = result.get('total_return_pct', 0)
            max_dd = result.get('max_drawdown_pct', 0)
            print(f"    夏普: {sharpe:.4f}, 总收益: {total_ret:.2f}%, 最大回撤: {max_dd:.2f}%")
            results.append(result)
        except Exception as e:
            print(f"    错误: {e}")
            traceback.print_exc()
    return results


def run_out_of_sample_test(extra_params, label):
    """样本外验证"""
    in_sample = run_backtest_with_params(
        extra_params=extra_params, label=f'{label}_is',
        start_date=TRAIN_START, end_date=TRAIN_END)
    out_sample = run_backtest_with_params(
        extra_params=extra_params, label=f'{label}_oos',
        start_date=VALID_START, end_date=VALID_END)
    baseline_is = run_backtest_with_params(
        extra_params=None, label='r2_baseline_is',
        start_date=TRAIN_START, end_date=TRAIN_END)
    baseline_oos = run_backtest_with_params(
        extra_params=None, label='r2_baseline_oos',
        start_date=VALID_START, end_date=VALID_END)

    is_imp = (in_sample.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
    oos_imp = (out_sample.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100
    return {
        'in_sample_sharpe': in_sample.get('sharpe_ratio', 0),
        'out_sample_sharpe': out_sample.get('sharpe_ratio', 0),
        'baseline_is_sharpe': baseline_is.get('sharpe_ratio', 0),
        'baseline_oos_sharpe': baseline_oos.get('sharpe_ratio', 0),
        'is_improvement_pct': is_imp,
        'oos_improvement_pct': oos_imp,
        'decay_ratio': oos_imp / is_imp if is_imp != 0 else 0,
    }


def run_parameter_sensitivity_test(param_name, param_value, label,
                                    perturbations=[-0.2, -0.1, 0.1, 0.2]):
    """参数敏感性测试"""
    results = {}
    base_result = run_backtest_with_params(
        extra_params={param_name: param_value}, label=f'{label}_base')
    for delta in perturbations:
        perturbed_value = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed_value = int(round(perturbed_value))
            if perturbed_value == 0:
                perturbed_value = 1
        result = run_backtest_with_params(
            extra_params={param_name: perturbed_value},
            label=f'{label}_perturb_{delta:+.0%}')
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed_value,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
        }
    base_sharpe = base_result.get('sharpe_ratio', 0)
    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_range = max(sharpe_values) - min(sharpe_values)
    return {
        'base_sharpe': base_sharpe,
        'base_param': param_value,
        'perturbation_results': results,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf'),
    }


def run_temporal_stability_test(extra_params, label, full_start=TRAIN_START, full_end=TRAIN_END):
    """时间稳定性测试"""
    from datetime import datetime
    start_year = datetime.strptime(full_start, '%Y-%m-%d').year
    end_year = datetime.strptime(full_end, '%Y-%m-%d').year
    yearly_results = []
    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < full_start:
            year_start = full_start
        if year_end > full_end:
            year_end = full_end
        opt_result = run_backtest_with_params(
            extra_params=extra_params, label=f'{label}_{year}',
            start_date=year_start, end_date=year_end)
        base_result = run_backtest_with_params(
            extra_params=None, label=f'r2_baseline_{year}',
            start_date=year_start, end_date=year_end)
        opt_sharpe = opt_result.get('sharpe_ratio', 0)
        base_sharpe = base_result.get('sharpe_ratio', 0)
        yearly_results.append({
            'year': year, 'opt_sharpe': opt_sharpe, 'base_sharpe': base_sharpe,
            'improvement': opt_sharpe - base_sharpe, 'is_positive': opt_sharpe - base_sharpe > 0,
        })
    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    return {
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': positive_years / total_years if total_years > 0 else 0,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'all', 'single', 'oos', 'sensitivity', 'temporal'], default='all')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--param-name', type=str, default=None)
    parser.add_argument('--param-value', type=float, default=None)
    args = parser.parse_args()

    if args.mode == 'baseline':
        print("运行基线回测...")
        result = run_backtest_with_params(label='r2_baseline')
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'all':
        results = run_all_optimizations()
        # 汇总
        print("\n" + "=" * 60)
        print("Round 2 优化结果汇总")
        print("=" * 60)
        baseline_sharpe = results[0].get('sharpe_ratio', 0) if results else 0
        print(f"{'优化方向':<25} {'夏普比率':>10} {'变化%':>8} {'总收益%':>10} {'最大回撤%':>10}")
        print("-" * 70)
        for r in results:
            label = r.get('label', '')
            name = next((n for l, n, p in R2_OPTIMIZATIONS if l == label), label)
            sharpe = r.get('sharpe_ratio', 0)
            ret = r.get('total_return_pct', 0)
            dd = r.get('max_drawdown_pct', 0)
            if label == 'r2_baseline':
                print(f"{name:<25} {sharpe:>10.4f} {'--':>8} {ret:>10.2f} {dd:>10.2f}")
            else:
                change = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if baseline_sharpe else 0
                print(f"{name:<25} {sharpe:>10.4f} {change:>+7.1f}% {ret:>10.2f} {dd:>10.2f}")

    elif args.mode == 'single':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in R2_OPTIMIZATIONS if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_backtest_with_params(extra_params=params, label=label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'oos':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in R2_OPTIMIZATIONS if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_out_of_sample_test(params, label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'sensitivity':
        if args.label is None or args.param_name is None or args.param_value is None:
            print("请指定 --label, --param-name, --param-value")
            sys.exit(1)
        result = run_parameter_sensitivity_test(args.param_name, args.param_value, args.label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'temporal':
        if args.label is None:
            print("请指定 --label")
            sys.exit(1)
        opt = next((o for o in R2_OPTIMIZATIONS if o[0] == args.label), None)
        if opt is None:
            print(f"未找到优化: {args.label}")
            sys.exit(1)
        label, name, params = opt
        result = run_temporal_stability_test(params, label)
        print(json.dumps(result, indent=2, ensure_ascii=False))

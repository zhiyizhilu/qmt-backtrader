"""样本外验证 + 参数敏感性 + 时间稳定性测试"""
import os
import sys
import json
import datetime
import traceback

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'arbr_small_cap'
STRATEGY_DIR = os.path.join(PROJECT_ROOT, 'strategies_my', 'arbr_small_cap_strategy')
RESULTS_DIR = os.path.join(STRATEGY_DIR, 'optimization', 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'
POOL = '中证1000'


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
        metrics['sharpe_ratio'] = sr
        metrics['total_return_pct'] = acc.rate * 100
        metrics['max_drawdown_pct'] = dd * 100
        metrics['label'] = label
        metrics['extra_params'] = extra_params
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def run_out_of_sample_test(label, extra_params):
    """样本外验证"""
    print(f'\n--- 样本外验证: {label} ---')

    # 测试集
    is_result = run_backtest_with_params(
        extra_params=extra_params, label=f'{label}_is',
        start_date=TRAIN_START, end_date=TRAIN_END)

    # 验证集
    oos_result = run_backtest_with_params(
        extra_params=extra_params, label=f'{label}_oos',
        start_date=VALID_START, end_date=VALID_END)

    # 基线对比
    baseline_is = run_backtest_with_params(
        extra_params=None, label='baseline_is',
        start_date=TRAIN_START, end_date=TRAIN_END)

    baseline_oos = run_backtest_with_params(
        extra_params=None, label='baseline_oos',
        start_date=VALID_START, end_date=VALID_END)

    is_sharpe = is_result.get('sharpe_ratio', 0)
    oos_sharpe = oos_result.get('sharpe_ratio', 0)
    baseline_is_sharpe = baseline_is.get('sharpe_ratio', 0)
    baseline_oos_sharpe = baseline_oos.get('sharpe_ratio', 0)

    is_improvement = (is_sharpe - baseline_is_sharpe) / abs(baseline_is_sharpe) * 100 if baseline_is_sharpe != 0 else 0
    oos_improvement = (oos_sharpe - baseline_oos_sharpe) / abs(baseline_oos_sharpe) * 100 if baseline_oos_sharpe != 0 else 0
    decay_ratio = oos_improvement / is_improvement if is_improvement != 0 else 0

    print(f'  IS: Sharpe={is_sharpe:.4f} (基线{baseline_is_sharpe:.4f}, 提升{is_improvement:+.1f}%)')
    print(f'  OOS: Sharpe={oos_sharpe:.4f} (基线{baseline_oos_sharpe:.4f}, 提升{oos_improvement:+.1f}%)')
    print(f'  衰减比: {decay_ratio:.2f}')

    return {
        'label': label,
        'is_sharpe': is_sharpe,
        'oos_sharpe': oos_sharpe,
        'baseline_is_sharpe': baseline_is_sharpe,
        'baseline_oos_sharpe': baseline_oos_sharpe,
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': decay_ratio,
    }


def run_parameter_sensitivity_test(label, param_name, param_value, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    """参数敏感性测试"""
    print(f'\n--- 参数敏感性: {label} ({param_name}={param_value}) ---')

    base_result = run_backtest_with_params(
        extra_params={param_name: param_value}, label=f'{label}_sens_base')
    base_sharpe = base_result.get('sharpe_ratio', 0)

    results = {}
    for delta in perturbations:
        perturbed_value = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed_value = int(round(perturbed_value))
            if perturbed_value == 0:
                perturbed_value = 1
        r = run_backtest_with_params(
            extra_params={param_name: perturbed_value},
            label=f'{label}_sens_{delta:+.0%}')
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed_value,
            'sharpe_ratio': r.get('sharpe_ratio', 0),
        }
        print(f'  {param_name}={perturbed_value}: Sharpe={r.get("sharpe_ratio", 0):.4f}')

    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_range = max(sharpe_values) - min(sharpe_values)
    sensitivity_ratio = sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf')

    print(f'  Base Sharpe={base_sharpe:.4f}, Range={sharpe_range:.4f}, Sensitivity={sensitivity_ratio:.2f}')

    return {
        'label': label,
        'base_sharpe': base_sharpe,
        'base_param': param_value,
        'perturbation_results': results,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': sensitivity_ratio,
    }


def run_temporal_stability_test(label, extra_params):
    """时间分段稳定性测试"""
    print(f'\n--- 时间稳定性: {label} ---')

    yearly_results = []
    for year in range(2020, 2025):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        if year_start < TRAIN_START:
            year_start = TRAIN_START
        if year_end > TRAIN_END:
            year_end = TRAIN_END

        opt_result = run_backtest_with_params(
            extra_params=extra_params, label=f'{label}_temp_{year}',
            start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
            extra_params=None, label=f'baseline_temp_{year}',
            start_date=year_start, end_date=year_end)

        opt_sharpe = opt_result.get('sharpe_ratio', 0)
        base_sharpe = base_result.get('sharpe_ratio', 0)
        improvement = opt_sharpe - base_sharpe

        yearly_results.append({
            'year': year,
            'opt_sharpe': opt_sharpe,
            'base_sharpe': base_sharpe,
            'improvement': improvement,
            'is_positive': improvement > 0,
        })
        print(f'  {year}: opt={opt_sharpe:.4f} base={base_sharpe:.4f} diff={improvement:+.4f} {"+" if improvement > 0 else "-"}')

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    print(f'  一致性: {positive_years}/{total_years} = {consistency_ratio:.2f}')

    return {
        'label': label,
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--oos', action='store_true', help='Run out-of-sample tests')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity tests')
    parser.add_argument('--temporal', action='store_true', help='Run temporal stability tests')
    parser.add_argument('--all', action='store_true', help='Run all review tests')
    args = parser.parse_args()

    effective_opts = [
        ('opt01_volatility_filter', '波动率过滤', {'max_volatility': 0.03}, 'max_volatility', 0.03),
        ('opt04_arbr_period_short', 'ARBR短周期', {'arbr_period': 14}, 'arbr_period', 14),
        ('opt07_turnover_control', '换手率控制', {'max_turnover_ratio': 0.5}, 'max_turnover_ratio', 0.5),
    ]

    review_results = []

    if args.oos or args.all:
        print('='*60)
        print('样本外验证')
        print('='*60)
        for label, name, params, _, _ in effective_opts:
            r = run_out_of_sample_test(label, params)
            review_results.append(r)

    if args.sensitivity or args.all:
        print('='*60)
        print('参数敏感性测试')
        print('='*60)
        for label, name, params, param_name, param_value in effective_opts:
            r = run_parameter_sensitivity_test(label, param_name, param_value)
            review_results.append(r)

    if args.temporal or args.all:
        print('='*60)
        print('时间稳定性测试')
        print('='*60)
        for label, name, params, _, _ in effective_opts:
            r = run_temporal_stability_test(label, params)
            review_results.append(r)

    # 保存审查汇总
    if review_results:
        summary_file = os.path.join(RESULTS_DIR, 'review_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(review_results, f, indent=2, ensure_ascii=False, default=str)
        print(f'\n审查结果已保存到: {summary_file}')

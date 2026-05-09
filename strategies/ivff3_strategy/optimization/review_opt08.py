import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'ivff3'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool='中证1000', start_date='2020-04-28', end_date='2026-04-28'):
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


def run_out_of_sample_test(strategy_name, extra_params, label,
                            full_start, full_end, split_ratio=0.5):
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(full_start, '%Y-%m-%d')
    end_dt = datetime.strptime(full_end, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    split_dt = start_dt + timedelta(days=int(total_days * split_ratio))
    split_date = split_dt.strftime('%Y-%m-%d')

    in_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_is', start_date=full_start, end_date=split_date)

    out_sample = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=extra_params,
        label=f'{label}_oos', start_date=split_date, end_date=full_end)

    baseline_is = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_is', start_date=full_start, end_date=split_date)

    baseline_oos = run_backtest_with_params(
        strategy_name=strategy_name, extra_params=None,
        label='baseline_oos', start_date=split_date, end_date=full_end)

    is_improvement = (in_sample.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
    oos_improvement = (out_sample.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100

    return {
        'in_sample_sharpe': in_sample.get('sharpe_ratio', 0),
        'out_sample_sharpe': out_sample.get('sharpe_ratio', 0),
        'baseline_is_sharpe': baseline_is.get('sharpe_ratio', 0),
        'baseline_oos_sharpe': baseline_oos.get('sharpe_ratio', 0),
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': oos_improvement / is_improvement if is_improvement != 0 else 0,
    }


def run_parameter_sensitivity_test(strategy_name, param_name, param_value,
                                    label, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    results = {}
    base_result = run_backtest_with_params(
        strategy_name=strategy_name,
        extra_params={param_name: param_value},
        label=f'{label}_base')

    for delta in perturbations:
        perturbed_value = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed_value = int(round(perturbed_value))
            if perturbed_value == 0:
                perturbed_value = 1
        result = run_backtest_with_params(
            strategy_name=strategy_name,
            extra_params={param_name: perturbed_value},
            label=f'{label}_perturb_{delta:+.0%}')
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed_value,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
        }

    base_sharpe = base_result.get('sharpe_ratio', 0)
    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_std = (sum((s - base_sharpe) ** 2 for s in sharpe_values) / len(sharpe_values)) ** 0.5
    sharpe_range = max(sharpe_values) - min(sharpe_values)

    return {
        'base_sharpe': base_sharpe,
        'base_param': param_value,
        'perturbation_results': results,
        'sharpe_std': sharpe_std,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf'),
    }


def run_temporal_stability_test(strategy_name, extra_params, label,
                                 full_start, full_end):
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
            strategy_name=strategy_name, extra_params=extra_params,
            label=f'{label}_{year}', start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
            strategy_name=strategy_name, extra_params=None,
            label=f'baseline_{year}', start_date=year_start, end_date=year_end)

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

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency_ratio = positive_years / total_years if total_years > 0 else 0

    return {
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': consistency_ratio,
    }


if __name__ == '__main__':
    print("="*60)
    print("样本外验证、参数敏感性、时间稳定性测试")
    print("="*60)
    
    # 1. 样本外验证
    print("\n1. 样本外验证测试...")
    oos_result = run_out_of_sample_test(
        'ivff3', {'max_iv_percentile': 0.3}, 'opt08_oos',
        '2020-04-28', '2026-04-28', 0.5)
    
    print(f"样本内夏普: {oos_result['in_sample_sharpe']:.4f}")
    print(f"样本外夏普: {oos_result['out_sample_sharpe']:.4f}")
    print(f"样本内改进: {oos_result['is_improvement_pct']:.1f}%")
    print(f"样本外改进: {oos_result['oos_improvement_pct']:.1f}%")
    print(f"衰减比: {oos_result['decay_ratio']:.2f}")
    
    # 2. 参数敏感性测试
    print("\n2. 参数敏感性测试...")
    sensitivity_result = run_parameter_sensitivity_test(
        'ivff3', 'max_iv_percentile', 0.3, 'opt08_sensitivity')
    
    print(f"基准夏普: {sensitivity_result['base_sharpe']:.4f}")
    print(f"参数范围: {sensitivity_result['sharpe_range']:.4f}")
    print(f"敏感度比率: {sensitivity_result['sensitivity_ratio']:.2f}")
    
    # 3. 时间稳定性测试
    print("\n3. 时间稳定性测试...")
    temporal_result = run_temporal_stability_test(
        'ivff3', {'max_iv_percentile': 0.3}, 'opt08_temporal',
        '2020-04-28', '2026-04-28')
    
    print(f"正改进年数: {temporal_result['positive_years']}/{temporal_result['total_years']}")
    print(f"一致性比率: {temporal_result['consistency_ratio']:.2f}")
    
    # 4. 综合评估
    print("\n" + "="*60)
    print("综合审查结论:")
    print("="*60)
    
    # 样本外判定
    if oos_result['decay_ratio'] >= 0.5:
        oos_verdict = "✅ 通过"
    elif oos_result['decay_ratio'] >= 0.2:
        oos_verdict = "⚠️ 有条件通过"
    else:
        oos_verdict = "❌ 不通过"
    
    # 参数敏感度判定
    if sensitivity_result['sensitivity_ratio'] < 0.3:
        sensitivity_verdict = "✅ 通过"
    elif sensitivity_result['sensitivity_ratio'] < 0.6:
        sensitivity_verdict = "⚠️ 有条件通过"
    else:
        sensitivity_verdict = "❌ 不通过"
    
    # 时间稳定性判定
    if temporal_result['consistency_ratio'] >= 0.7:
        temporal_verdict = "✅ 通过"
    elif temporal_result['consistency_ratio'] >= 0.5:
        temporal_verdict = "⚠️ 有条件通过"
    else:
        temporal_verdict = "❌ 不通过"
    
    print(f"样本外衰减比: {oos_result['decay_ratio']:.2f} - {oos_verdict}")
    print(f"参数敏感度: {sensitivity_result['sensitivity_ratio']:.2f} - {sensitivity_verdict}")
    print(f"时间稳定性: {temporal_result['consistency_ratio']:.2f} - {temporal_verdict}")
    
    # 最终结论
    if oos_verdict.startswith("✅") and sensitivity_verdict.startswith("✅") and temporal_verdict.startswith("✅"):
        final_verdict = "✅ 强力通过，优先组合"
    elif oos_verdict.startswith("✅") or sensitivity_verdict.startswith("✅") or temporal_verdict.startswith("✅"):
        final_verdict = "⚠️ 有条件通过，组合时降低权重"
    else:
        final_verdict = "❌ 不通过，放弃该优化"
    
    print(f"\n最终结论: {final_verdict}")
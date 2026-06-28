"""硬逻辑与过度拟合审查脚本 - 样本外验证、参数敏感性、时间稳定性"""
import os
import sys
import json
import datetime
import subprocess
import glob
import time

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from strategies import get_strategy_dir

STRATEGY_NAME = 'etf_momentum_epo'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON_EXE = os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe')
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'


def run_backtest(extra_params=None, label='test', start_date='2020-04-28', end_date='2024-04-28'):
    """运行单次回测"""
    cmd = [sys.executable, MAIN_SCRIPT, '--strategy', STRATEGY_NAME,
           '--start', start_date, '--end', end_date, '--ai-mode']

    if extra_params:
        param_parts = []
        for k, v in extra_params.items():
            if isinstance(v, bool):
                param_parts.append(f'{k}={str(v).lower()}')
            elif isinstance(v, (int, float)):
                param_parts.append(f'{k}={v}')
            else:
                param_parts.append(f'{k}={v}')
        cmd.extend(['--strategy-params', ','.join(param_parts)])

    print(f'  Running: {label} ({start_date}~{end_date}) params={extra_params}')
    sys.stdout.flush()

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=1800, env={**os.environ, 'QMT_LOG_LEVEL': 'WARNING'})
    except Exception as e:
        return {'label': label, 'error': str(e)}

    # 查找最新结果
    results_dir = os.path.join(STRATEGY_DIR, 'backtest_results')
    files = glob.glob(os.path.join(results_dir, f'*_{STRATEGY_NAME}.json'))
    if not files:
        return {'label': label, 'error': 'No result file'}
    result_path = max(files, key=os.path.getmtime)
    time.sleep(1)

    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        m = data.get('metrics', {})
        metrics = {
            'label': label,
            'sharpe_ratio': m.get('sharpe_ratio', 0),
            'total_return_pct': m.get('total_return_pct', 0),
            'max_drawdown_pct': m.get('max_drawdown_pct', 0),
            'annual_return_pct': m.get('annual_return_pct', 0),
            'extra_params': extra_params,
            'timestamp': datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        metrics = {'label': label, 'error': str(e)}

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f'  => sharpe={metrics.get("sharpe_ratio", "N/A")}')
    return metrics


def run_out_of_sample_test(label, extra_params):
    """样本外验证"""
    print(f'\n{"="*60}')
    print(f'OOS Test: {label}')
    print(f'{"="*60}')

    is_result = run_backtest(extra_params=extra_params, label=f'{label}_is',
                              start_date=TRAIN_START, end_date=TRAIN_END)
    oos_result = run_backtest(extra_params=extra_params, label=f'{label}_oos',
                               start_date=VALID_START, end_date=VALID_END)
    baseline_is = run_backtest(extra_params=None, label='review_baseline_is',
                                start_date=TRAIN_START, end_date=TRAIN_END)
    baseline_oos = run_backtest(extra_params=None, label='review_baseline_oos',
                                 start_date=VALID_START, end_date=VALID_END)

    is_sharpe = is_result.get('sharpe_ratio', 0)
    oos_sharpe = oos_result.get('sharpe_ratio', 0)
    b_is_sharpe = baseline_is.get('sharpe_ratio', 0)
    b_oos_sharpe = baseline_oos.get('sharpe_ratio', 0)

    is_imp = (is_sharpe - b_is_sharpe) / abs(b_is_sharpe) * 100 if b_is_sharpe != 0 else 0
    oos_imp = (oos_sharpe - b_oos_sharpe) / abs(b_oos_sharpe) * 100 if b_oos_sharpe != 0 else 0

    decay_ratio = oos_imp / is_imp if is_imp != 0 else 0

    result = {
        'label': label,
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'valid_period': f'{VALID_START} ~ {VALID_END}',
        'in_sample_sharpe': is_sharpe,
        'out_sample_sharpe': oos_sharpe,
        'baseline_is_sharpe': b_is_sharpe,
        'baseline_oos_sharpe': b_oos_sharpe,
        'is_improvement_pct': is_imp,
        'oos_improvement_pct': oos_imp,
        'decay_ratio': decay_ratio,
    }

    print(f'  IS: {is_sharpe:.4f} (baseline {b_is_sharpe:.4f}, improvement {is_imp:+.1f}%)')
    print(f'  OOS: {oos_sharpe:.4f} (baseline {b_oos_sharpe:.4f}, improvement {oos_imp:+.1f}%)')
    print(f'  Decay ratio: {decay_ratio:.2f}')

    return result


def run_parameter_sensitivity_test(label, param_name, param_value, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    """参数敏感性分析"""
    print(f'\n{"="*60}')
    print(f'Sensitivity Test: {label} ({param_name}={param_value})')
    print(f'{"="*60}')

    base_result = run_backtest(extra_params={param_name: param_value},
                                label=f'{label}_sens_base',
                                start_date=TRAIN_START, end_date=TRAIN_END)
    base_sharpe = base_result.get('sharpe_ratio', 0)

    results = {}
    for delta in perturbations:
        perturbed = param_value * (1 + delta)
        if isinstance(param_value, int):
            perturbed = int(round(perturbed))
            if perturbed == 0:
                perturbed = 1
        if isinstance(param_value, float):
            perturbed = round(perturbed, 6)

        r = run_backtest(extra_params={param_name: perturbed},
                          label=f'{label}_sens_{delta:+.0%}',
                          start_date=TRAIN_START, end_date=TRAIN_END)
        results[f'perturb_{delta:+.0%}'] = {
            'param_value': perturbed,
            'sharpe_ratio': r.get('sharpe_ratio', 0),
        }

    sharpe_values = [r['sharpe_ratio'] for r in results.values()]
    sharpe_range = max(sharpe_values) - min(sharpe_values) if sharpe_values else 0
    sensitivity_ratio = sharpe_range / abs(base_sharpe) if base_sharpe != 0 else float('inf')

    result = {
        'label': label,
        'param_name': param_name,
        'base_param': param_value,
        'base_sharpe': base_sharpe,
        'perturbation_results': results,
        'sharpe_range': sharpe_range,
        'sensitivity_ratio': round(sensitivity_ratio, 4),
    }

    print(f'  Base sharpe: {base_sharpe:.4f}')
    print(f'  Range: {sharpe_range:.4f}')
    print(f'  Sensitivity ratio: {sensitivity_ratio:.4f}')

    return result


def run_temporal_stability_test(label, extra_params):
    """时间分段稳定性测试"""
    print(f'\n{"="*60}')
    print(f'Temporal Stability Test: {label}')
    print(f'{"="*60}')

    years = [(2020, '2020-04-28', '2020-12-31'),
             (2021, '2021-01-01', '2021-12-31'),
             (2022, '2022-01-01', '2022-12-31'),
             (2023, '2023-01-01', '2023-12-31'),
             (2024, '2024-01-01', '2024-04-28')]

    yearly_results = []
    for year, start, end in years:
        opt_r = run_backtest(extra_params=extra_params,
                              label=f'{label}_year_{year}',
                              start_date=start, end_date=end)
        base_r = run_backtest(extra_params=None,
                               label=f'baseline_year_{year}',
                               start_date=start, end_date=end)

        opt_sharpe = opt_r.get('sharpe_ratio', 0)
        base_sharpe = base_r.get('sharpe_ratio', 0)
        improvement = opt_sharpe - base_sharpe

        yearly_results.append({
            'year': year,
            'opt_sharpe': opt_sharpe,
            'base_sharpe': base_sharpe,
            'improvement': round(improvement, 4),
            'is_positive': improvement > 0,
        })
        print(f'  {year}: opt={opt_sharpe:.4f} base={base_sharpe:.4f} delta={improvement:+.4f} {"✓" if improvement > 0 else "✗"}')

    positive_years = sum(1 for r in yearly_results if r['is_positive'])
    total_years = len(yearly_results)
    consistency = positive_years / total_years if total_years > 0 else 0

    result = {
        'label': label,
        'yearly_results': yearly_results,
        'positive_years': positive_years,
        'total_years': total_years,
        'consistency_ratio': round(consistency, 2),
    }

    print(f'  Consistency: {positive_years}/{total_years} = {consistency:.0%}')

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['oos', 'sensitivity', 'temporal', 'all'], default='all')
    args = parser.parse_args()

    # 有效优化参数
    effective_opts = [
        ('opt01_volatility_filter', {'max_volatility': 0.03}, 'max_volatility', 0.03),
        ('opt03_min_r_squared', {'min_r_squared': 0.2}, 'min_r_squared', 0.2),
        ('opt07_min_score', {'min_score': 0.05}, 'min_score', 0.05),
    ]

    review_results = {}

    for label, params, param_name, param_value in effective_opts:
        review_results[label] = {}

        if args.test in ('oos', 'all'):
            oos = run_out_of_sample_test(label, params)
            review_results[label]['oos'] = oos

        if args.test in ('sensitivity', 'all'):
            sens = run_parameter_sensitivity_test(label, param_name, param_value)
            review_results[label]['sensitivity'] = sens

        if args.test in ('temporal', 'all'):
            temporal = run_temporal_stability_test(label, params)
            review_results[label]['temporal'] = temporal

    # 保存审查汇总
    summary_file = os.path.join(RESULTS_DIR, 'review_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(review_results, f, indent=2, ensure_ascii=False)

    # 输出综合结论
    print(f'\n{"="*80}')
    print('综合审查结论')
    print(f'{"="*80}')

    for label, params, param_name, param_value in effective_opts:
        r = review_results.get(label, {})
        oos = r.get('oos', {})
        sens = r.get('sensitivity', {})
        temp = r.get('temporal', {})

        decay = oos.get('decay_ratio', 0)
        sens_ratio = sens.get('sensitivity_ratio', 0)
        consistency = temp.get('consistency_ratio', 0)

        # 判定
        if decay >= 0.5 and sens_ratio < 0.3 and consistency >= 0.7:
            conclusion = '✅ 通过'
        elif decay >= 0.2 and sens_ratio < 0.6 and consistency >= 0.5:
            conclusion = '⚠️ 有条件通过'
        else:
            conclusion = '❌ 不通过'

        print(f'\n{label}:')
        print(f'  样本外衰减比: {decay:.2f} ({"通过" if decay >= 0.5 else "有条件" if decay >= 0.2 else "不通过"})')
        print(f'  参数敏感度: {sens_ratio:.4f} ({"通过" if sens_ratio < 0.3 else "有条件" if sens_ratio < 0.6 else "不通过"})')
        print(f'  时间稳定性: {consistency:.0%} ({"通过" if consistency >= 0.7 else "有条件" if consistency >= 0.5 else "不通过"})')
        print(f'  综合结论: {conclusion}')

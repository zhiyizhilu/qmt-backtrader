"""v2c blend精调回测"""
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
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'v2c_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

V2C_TESTS = [
    ('baseline', '基线策略', {}),
    ('blend_035', 'blend=0.35', {'risk_parity_blend': 0.35}),
    ('blend_040', 'blend=0.40', {'risk_parity_blend': 0.40}),
    ('blend_045', 'blend=0.45', {'risk_parity_blend': 0.45}),
    ('blend_050', 'blend=0.50', {'risk_parity_blend': 0.50}),
    ('blend_055', 'blend=0.55', {'risk_parity_blend': 0.55}),
    ('blend_060', 'blend=0.60', {'risk_parity_blend': 0.60}),
    ('blend_065', 'blend=0.65', {'risk_parity_blend': 0.65}),
]


def run_backtest(extra_params=None, label='test', start_date='2020-04-28', end_date='2024-04-28'):
    cmd = [sys.executable, MAIN_SCRIPT, '--strategy', STRATEGY_NAME,
           '--start', start_date, '--end', end_date, '--ai-mode']
    if extra_params:
        param_parts = []
        for k, v in extra_params.items():
            param_parts.append(f'{k}={v}')
        cmd.extend(['--strategy-params', ','.join(param_parts)])
    print(f'  Running: {label} ({start_date}~{end_date})')
    sys.stdout.flush()
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=1800, env={**os.environ, 'QMT_LOG_LEVEL': 'WARNING'})
    except Exception as e:
        return {'label': label, 'error': str(e)}
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
            'label': label, 'sharpe_ratio': m.get('sharpe_ratio', 0),
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
    sharpe = metrics.get('sharpe_ratio', 'N/A')
    ret = metrics.get('total_return_pct', 'N/A')
    print(f'  => sharpe={sharpe} return={ret}%')
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['is', 'oos', 'all'], default='all')
    args = parser.parse_args()

    is_results = []
    oos_results = []

    if args.phase in ('is', 'all'):
        print('='*60)
        print('v2c blend精调 - 样本内')
        print('='*60)
        for label, name, params in V2C_TESTS:
            print(f'\n--- {name} ---')
            r = run_backtest(extra_params=params if params else None,
                              label=f'{label}_is', start_date=TRAIN_START, end_date=TRAIN_END)
            r['name'] = name; r['label_key'] = label
            is_results.append(r)

    if args.phase in ('oos', 'all'):
        print('\n' + '='*60)
        print('v2c blend精调 - 样本外')
        print('='*60)
        for label, name, params in V2C_TESTS:
            print(f'\n--- {name} ---')
            r = run_backtest(extra_params=params if params else None,
                              label=f'{label}_oos', start_date=VALID_START, end_date=VALID_END)
            r['name'] = name; r['label_key'] = label
            oos_results.append(r)

    b_is = next((r.get('sharpe_ratio', 0) for r in is_results if r.get('label_key') == 'baseline'), 0)
    b_oos = next((r.get('sharpe_ratio', 0) for r in oos_results if r.get('label_key') == 'baseline'), 0)

    print(f'\n{"="*100}')
    print(f'v2c blend精调结果 (基线 IS={b_is:.4f} OOS={b_oos:.4f})')
    print(f'{"─"*100}')
    print(f'{"方案":<20} {"IS夏普":>8} {"OOS夏普":>8} {"IS变化":>8} {"OOS变化":>8} {"IS收益%":>10} {"OOS收益%":>10} {"IS最大回撤%":>12} {"OOS最大回撤%":>12}')
    print(f'{"─"*100}')

    for i, (label, name, params) in enumerate(V2C_TESTS):
        is_r = is_results[i] if i < len(is_results) else {}
        oos_r = oos_results[i] if i < len(oos_results) else {}
        is_s = is_r.get('sharpe_ratio', 0)
        oos_s = oos_r.get('sharpe_ratio', 0)
        is_ret = is_r.get('total_return_pct', 0)
        oos_ret = oos_r.get('total_return_pct', 0)
        is_dd = is_r.get('max_drawdown_pct', 0)
        oos_dd = oos_r.get('max_drawdown_pct', 0)
        is_imp = (is_s - b_is) / abs(b_is) * 100 if b_is != 0 else 0
        oos_imp = (oos_s - b_oos) / abs(b_oos) * 100 if b_oos != 0 else 0
        flag = ' ***' if oos_imp > 5 and is_imp > -5 else ' **' if oos_imp > 0 and is_imp > -5 else ''
        print(f'{name:<20} {is_s:>8.4f} {oos_s:>8.4f} {is_imp:>+7.1f}% {oos_imp:>+7.1f}% {is_ret:>9.1f}% {oos_ret:>9.1f}% {is_dd:>11.1f}% {oos_dd:>11.1f}%{flag}')

    summary = {'baseline_is_sharpe': b_is, 'baseline_oos_sharpe': b_oos,
               'is_results': is_results, 'oos_results': oos_results}
    with open(os.path.join(RESULTS_DIR, 'v2c_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

"""组合优化回测脚本"""
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

MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# 组合优化方案
COMBINED_TESTS = [
    ('combo01_vol_r2', '波动率(0.03)+R²(0.2)', {'max_volatility': 0.03, 'min_r_squared': 0.2}),
    ('combo02_vol_r2_score', '波动率(0.03)+R²(0.2)+分数(0.05)', {'max_volatility': 0.03, 'min_r_squared': 0.2, 'min_score': 0.05}),
    ('combo03_vol002', '波动率(0.02)', {'max_volatility': 0.02}),
    ('combo04_vol002_r2', '波动率(0.02)+R²(0.2)', {'max_volatility': 0.02, 'min_r_squared': 0.2}),
    ('combo05_vol004', '波动率(0.04)', {'max_volatility': 0.04}),
]


def run_backtest(extra_params=None, label='test', start_date='2020-04-28', end_date='2024-04-28'):
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

    print(f'  => sharpe={metrics.get("sharpe_ratio", "N/A")} return={metrics.get("total_return_pct", "N/A")}%')
    return metrics


if __name__ == '__main__':
    # 测试集回测
    print('='*60)
    print('组合优化 - 测试集 (IS)')
    print('='*60)

    is_results = []
    for label, name, params in COMBINED_TESTS:
        print(f'\n--- {name} ---')
        r = run_backtest(extra_params=params, label=f'{label}_is',
                          start_date=TRAIN_START, end_date=TRAIN_END)
        r['name'] = name
        is_results.append(r)

    # 验证集回测
    print('\n' + '='*60)
    print('组合优化 - 验证集 (OOS)')
    print('='*60)

    oos_results = []
    for label, name, params in COMBINED_TESTS:
        print(f'\n--- {name} ---')
        r = run_backtest(extra_params=params, label=f'{label}_oos',
                          start_date=VALID_START, end_date=VALID_END)
        r['name'] = name
        oos_results.append(r)

    # 基线
    print('\n--- Baseline IS ---')
    baseline_is = run_backtest(extra_params=None, label='combo_baseline_is',
                                start_date=TRAIN_START, end_date=TRAIN_END)
    print('--- Baseline OOS ---')
    baseline_oos = run_backtest(extra_params=None, label='combo_baseline_oos',
                                 start_date=VALID_START, end_date=VALID_END)

    # 汇总
    b_is_sharpe = baseline_is.get('sharpe_ratio', 0)
    b_oos_sharpe = baseline_oos.get('sharpe_ratio', 0)

    print(f'\n{"="*80}')
    print('组合优化结果汇总')
    print(f'基线 IS夏普={b_is_sharpe:.4f} OOS夏普={b_oos_sharpe:.4f}')
    print(f'{"─"*80}')
    print(f'{"方案":<30} {"IS夏普":>8} {"IS变化":>8} {"OOS夏普":>8} {"OOS变化":>8} {"衰减比":>8}')
    print(f'{"─"*80}')

    for i, (label, name, params) in enumerate(COMBINED_TESTS):
        is_r = is_results[i]
        oos_r = oos_results[i]
        is_sharpe = is_r.get('sharpe_ratio', 0)
        oos_sharpe = oos_r.get('sharpe_ratio', 0)
        is_imp = (is_sharpe - b_is_sharpe) / abs(b_is_sharpe) * 100 if b_is_sharpe != 0 else 0
        oos_imp = (oos_sharpe - b_oos_sharpe) / abs(b_oos_sharpe) * 100 if b_oos_sharpe != 0 else 0
        decay = oos_imp / is_imp if is_imp != 0 else 0
        print(f'{name:<30} {is_sharpe:>8.4f} {is_imp:>+7.1f}% {oos_sharpe:>8.4f} {oos_imp:>+7.1f}% {decay:>8.2f}')

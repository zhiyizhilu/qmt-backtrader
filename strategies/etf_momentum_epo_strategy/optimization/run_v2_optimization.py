"""v2优化回测脚本 - 全新维度优化方向"""
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
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'v2_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# v2优化方案
V2_OPTIMIZATIONS = [
    ('baseline', '基线策略', {}),
    # 动量信号改进
    ('v2A_multi_period', '多周期融合(0.3/0.4/0.3)', {'multi_period_weights': '0.3,0.4,0.3'}),
    ('v2A_multi_period_02060', '多周期融合(0.2/0.6/0.2)', {'multi_period_weights': '0.2,0.6,0.2'}),
    ('v2B_accel_05', '动量加速度(bonus=0.5)', {'momentum_accel_bonus': 0.5}),
    ('v2B_accel_10', '动量加速度(bonus=1.0)', {'momentum_accel_bonus': 1.0}),
    ('v2C_volume_confirm', '成交量确认动量', {'volume_confirm': True}),
    # 择时与仓位管理
    ('v2D_breadth', '市场宽度仓位', {'market_breadth_sizing': True}),
    ('v2E_strength_010', '趋势强度仓位(ref=0.10)', {'trend_strength_ref': 0.10}),
    ('v2E_strength_020', '趋势强度仓位(ref=0.20)', {'trend_strength_ref': 0.20}),
    # 组合构建优化
    ('v2F_corr_07', '低相关性优选(corr>0.7替换)', {'correlation_threshold': 0.7}),
    ('v2F_corr_08', '低相关性优选(corr>0.8替换)', {'correlation_threshold': 0.8}),
    ('v2G_risk_parity', '风险平价权重', {'use_risk_parity': True}),
    # 调仓机制改进
    ('v2H_skip_stable', '排名稳定跳过调仓', {'skip_stable_rank': True}),
    ('v2I_quarterly', '季频调仓', {'quarterly_rebalance': True}),
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

    # === 样本内回测 ===
    if args.phase in ('is', 'all'):
        print('='*60)
        print('v2优化 - 样本内回测 (IS: 2020-04-28 ~ 2024-04-28)')
        print('='*60)

        for label, name, params in V2_OPTIMIZATIONS:
            print(f'\n--- {name} ---')
            r = run_backtest(extra_params=params if params else None,
                              label=f'{label}_is',
                              start_date=TRAIN_START, end_date=TRAIN_END)
            r['name'] = name
            r['label_key'] = label
            is_results.append(r)

    # === 样本外回测 ===
    if args.phase in ('oos', 'all'):
        print('\n' + '='*60)
        print('v2优化 - 样本外回测 (OOS: 2024-04-28 ~ 2026-04-28)')
        print('='*60)

        for label, name, params in V2_OPTIMIZATIONS:
            print(f'\n--- {name} ---')
            r = run_backtest(extra_params=params if params else None,
                              label=f'{label}_oos',
                              start_date=VALID_START, end_date=VALID_END)
            r['name'] = name
            r['label_key'] = label
            oos_results.append(r)

    # === 汇总 ===
    b_is_sharpe = next((r.get('sharpe_ratio', 0) for r in is_results if r.get('label_key') == 'baseline'), 0)
    b_oos_sharpe = next((r.get('sharpe_ratio', 0) for r in oos_results if r.get('label_key') == 'baseline'), 0)

    print(f'\n{"="*100}')
    print(f'v2优化结果汇总')
    print(f'基线 IS夏普={b_is_sharpe:.4f} OOS夏普={b_oos_sharpe:.4f}')
    print(f'{"─"*100}')
    print(f'{"方案":<30} {"IS夏普":>8} {"IS变化":>8} {"OOS夏普":>8} {"OOS变化":>8} {"衰减比":>8}')
    print(f'{"─"*100}')

    for i, (label, name, params) in enumerate(V2_OPTIMIZATIONS):
        is_r = is_results[i] if i < len(is_results) else {}
        oos_r = oos_results[i] if i < len(oos_results) else {}
        is_sharpe = is_r.get('sharpe_ratio', 0)
        oos_sharpe = oos_r.get('sharpe_ratio', 0)
        is_imp = (is_sharpe - b_is_sharpe) / abs(b_is_sharpe) * 100 if b_is_sharpe != 0 else 0
        oos_imp = (oos_sharpe - b_oos_sharpe) / abs(b_oos_sharpe) * 100 if b_oos_sharpe != 0 else 0
        decay = oos_imp / is_imp if is_imp != 0 else 0

        flag = ''
        if is_imp >= 5 and oos_imp >= 0:
            flag = ' ***'
        elif is_imp >= 5 and oos_imp >= -10:
            flag = ' **'
        elif is_imp >= 5:
            flag = ' *'

        print(f'{name:<30} {is_sharpe:>8.4f} {is_imp:>+7.1f}% {oos_sharpe:>8.4f} {oos_imp:>+7.1f}% {decay:>8.2f}{flag}')

    # 保存汇总
    summary = {
        'baseline_is_sharpe': b_is_sharpe,
        'baseline_oos_sharpe': b_oos_sharpe,
        'is_results': [{k: v for k, v in r.items() if k != 'error'} for r in is_results],
        'oos_results': [{k: v for k, v in r.items() if k != 'error'} for r in oos_results],
    }
    with open(os.path.join(RESULTS_DIR, 'v2_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

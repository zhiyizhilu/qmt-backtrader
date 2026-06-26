"""样本外验证脚本 - 对有效优化进行验证集回测"""
import os
import sys
import json
import subprocess
import glob
import time
import datetime

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from strategies import get_strategy_dir

STRATEGY_NAME = 'all_weather_rotation'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# 有效优化方案
EFFECTIVE_OPTS = [
    ('opt01_volatility_filter', '波动率过滤 (vol=0.02)', {'max_volatility': 0.02}),
    ('opt04_switch_threshold', '换仓阈值 (5%)', {'switch_threshold': 0.05}),
    ('opt05_relaxed_small', '放宽SMALL (ROE>8%,ROA>5%)', {'small_min_roe_relaxed': 0.08, 'small_min_roa_relaxed': 0.05}),
    ('opt07_long_momentum', '20日动量', {'momentum_days_long': 20}),
    ('opt09_dual_period', '双周期动量验证', {'dual_period_momentum': True}),
]


def run_oos_backtest(extra_params, label):
    cmd = [sys.executable, MAIN_SCRIPT, '--strategy', STRATEGY_NAME,
           '--pool', '沪深300', '--start', VALID_START, '--end', VALID_END, '--ai-mode']
    if extra_params:
        parts = []
        for k, v in extra_params.items():
            if isinstance(v, bool):
                parts.append(f'{k}={str(v).lower()}')
            else:
                parts.append(f'{k}={v}')
        cmd.extend(['--strategy-params', ','.join(parts)])

    print(f'\n>>> OOS: {label}')
    results_dir = os.path.join(STRATEGY_DIR, 'backtest_results')
    before = set(glob.glob(os.path.join(results_dir, '*.json')))
    before_time = time.time()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800,
                           env={**os.environ, 'QMT_LOG_LEVEL': 'WARNING'})

    # 查找新结果文件
    result_path = None
    for _ in range(10):
        time.sleep(1)
        after = set(glob.glob(os.path.join(results_dir, '*.json')))
        new = [f for f in after if f not in before or os.path.getmtime(f) >= before_time]
        if new:
            result_path = max(new, key=os.path.getmtime)
            break

    if not result_path:
        all_files = glob.glob(os.path.join(results_dir, '*.json'))
        if all_files:
            result_path = max(all_files, key=os.path.getmtime)

    if not result_path:
        return {'label': label, 'error': 'no result file'}

    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    m = data.get('metrics', {})
    metrics = {
        'label': label,
        'sharpe_ratio': m.get('sharpe_ratio', 0),
        'total_return_pct': m.get('total_return_pct', 0),
        'max_drawdown_pct': m.get('max_drawdown_pct', 0),
        'annual_return_pct': m.get('annual_return_pct', 0),
    }
    print(f'  Sharpe: {metrics["sharpe_ratio"]:.4f}, Return: {metrics["total_return_pct"]:.2f}%, DD: {metrics["max_drawdown_pct"]:.2f}%')
    return metrics


if __name__ == '__main__':
    # 先跑baseline
    print("=" * 60)
    print("样本外验证 - 验证集: 2024-04-28 ~ 2026-04-28")
    print("=" * 60)

    baseline = run_oos_backtest(None, 'baseline_oos')
    baseline_sharpe = baseline.get('sharpe_ratio', 0)

    results = [('baseline', '基线', baseline_sharpe, 0)]
    oos_data = {'baseline': baseline}

    for label, name, params in EFFECTIVE_OPTS:
        r = run_oos_backtest(params, f'{label}_oos')
        oos_sharpe = r.get('sharpe_ratio', 0)
        improvement = (oos_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        results.append((label, name, oos_sharpe, improvement))
        oos_data[label] = r

    # 读取训练集结果
    train_data = {}
    for label, _, _ in EFFECTIVE_OPTS:
        fpath = os.path.join(RESULTS_DIR, f'{label}.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                train_data[label] = json.load(f)

    baseline_train_path = os.path.join(RESULTS_DIR, 'baseline.json')
    if os.path.exists(baseline_train_path):
        with open(baseline_train_path, 'r', encoding='utf-8') as f:
            train_data['baseline'] = json.load(f)

    # 打印对比表
    print("\n" + "=" * 80)
    print("样本内 vs 样本外 对比")
    print("=" * 80)
    base_is = train_data.get('baseline', {}).get('sharpe_ratio', 0)
    base_oos = baseline_sharpe

    print(f"{'方案':<25} {'训练集夏普':>10} {'验证集夏普':>10} {'IS提升':>8} {'OOS提升':>8} {'衰减比':>8} {'判定':>6}")
    print("-" * 80)

    for label, name, oos_sharpe, oos_improvement in results:
        if label == 'baseline':
            is_sharpe = base_is
            is_imp = 0
            oos_imp = 0
            decay = '-'
        else:
            is_sharpe = train_data.get(label, {}).get('sharpe_ratio', 0)
            is_imp = (is_sharpe - base_is) / abs(base_is) * 100 if base_is != 0 else 0
            oos_imp = oos_improvement
            decay = f'{oos_imp/is_imp:.2f}' if is_imp != 0 else '-'

        # 判定：衰减比>0.3且OOS提升>0则通过
        if label == 'baseline':
            verdict = '-'
        else:
            try:
                decay_val = oos_imp / is_imp if is_imp != 0 else 0
            except:
                decay_val = 0
            verdict = '通过' if decay_val > 0.3 and oos_imp > 0 else '过拟合'

        print(f"{name:<25} {is_sharpe:>10.4f} {oos_sharpe:>10.4f} {is_imp:>+7.1f}% {oos_imp:>+7.1f}% {decay:>8} {verdict:>6}")

    # 保存结果
    oos_file = os.path.join(RESULTS_DIR, 'oos_validation.json')
    with open(oos_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_oos': baseline,
            'oos_results': oos_data,
            'train_results': train_data,
            'timestamp': datetime.datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {oos_file}")

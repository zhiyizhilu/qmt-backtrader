"""ETF动量EPO策略优化脚本 - 通过命令行回测

通过 subprocess 调用 main.py 运行回测，从结果JSON中提取指标。
"""
import os
import sys
import json
import datetime
import subprocess
import glob
import time
import traceback

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

# 固定的测试集/验证集时间边界
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# 所有优化方案
ALL_OPTIMIZATIONS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤 (vol=0.03)', {'max_volatility': 0.03}),
    ('opt02_switch_threshold', '换仓阈值 (5%)', {'switch_threshold': 0.05}),
    ('opt03_min_r_squared', 'R²最小阈值 (0.2)', {'min_r_squared': 0.2}),
    ('opt04_momentum_days_20', '动量天数 (20)', {'momentum_days_opt': 20}),
    ('opt05_top_n_5', '持仓数量 (5)', {'top_n_opt': 5}),
    ('opt06_max_weight', '最大持仓权重 (40%)', {'max_weight': 0.4}),
    ('opt07_min_score', '动量分数最低门槛 (0.05)', {'min_score': 0.05}),
    ('opt08_epo_w_05', 'EPO收缩权重 (0.5)', {'epo_w_opt': 0.5}),
    ('opt09_cash_etf', '空仓转货币ETF', {'cash_etf_code': '511880.SH'}),
    ('opt10_bimonthly', '双月调仓', {'rebalance_freq': 2}),
]


def _find_latest_result(strategy_dir, strategy_name, min_age_seconds=2):
    """查找最新的回测结果JSON文件"""
    results_dir = os.path.join(strategy_dir, 'backtest_results')
    pattern = os.path.join(results_dir, f'*_{strategy_name}.json')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    for f in files:
        if time.time() - os.path.getmtime(f) > min_age_seconds:
            return f
    return files[0]


def _extract_metrics_from_result(result_path, label, extra_params=None):
    """从回测结果JSON中提取关键指标"""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {'label': label, 'error': f'读取结果失败: {e}'}

    metrics = {'label': label, 'extra_params': extra_params}

    m = data.get('metrics', {})
    if m:
        metrics['initial_capital'] = m.get('initial_capital', 0)
        metrics['final_value'] = m.get('final_value', 0)
        metrics['total_return_pct'] = m.get('total_return_pct', 0)
        metrics['sharpe_ratio'] = m.get('sharpe_ratio', 0)
        metrics['max_drawdown_pct'] = m.get('max_drawdown_pct', m.get('max_drawdown', 0))
        metrics['annual_return_pct'] = m.get('annual_return_pct', 0)
        metrics['trading_days'] = m.get('total_trading_days', m.get('trading_days', 0))
        metrics['fee'] = m.get('fee', 0)
        metrics['turnover'] = m.get('turnover', 0)

    metrics.setdefault('sharpe_ratio', data.get('sharpe_ratio', 0))
    metrics.setdefault('max_drawdown_pct', data.get('max_drawdown_pct', data.get('max_drawdown', 0)))
    metrics.setdefault('total_return_pct', data.get('total_return_pct', 0))
    metrics.setdefault('annual_return_pct', data.get('annual_return_pct', 0))

    metrics['timestamp'] = datetime.datetime.now().isoformat()
    return metrics


def run_backtest_with_params(extra_params=None, label='test',
                              start_date='2020-04-28', end_date='2024-04-28'):
    """通过命令行运行回测并提取结果"""
    cmd = [
        sys.executable, MAIN_SCRIPT,
        '--strategy', STRATEGY_NAME,
        '--start', start_date,
        '--end', end_date,
        '--ai-mode',
    ]

    if extra_params:
        param_parts = []
        for k, v in extra_params.items():
            if isinstance(v, bool):
                param_parts.append(f'{k}={str(v).lower()}')
            elif isinstance(v, (int, float)):
                param_parts.append(f'{k}={v}')
            elif isinstance(v, str):
                param_parts.append(f'{k}={v}')
        params_str = ','.join(param_parts)
        cmd.extend(['--strategy-params', params_str])

    print(f'\n{"="*60}')
    print(f'Running backtest: {label}')
    print(f'Extra params: {extra_params}')
    print(f'{"="*60}')
    sys.stdout.flush()

    results_dir = os.path.join(STRATEGY_DIR, 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    before_files = set(glob.glob(os.path.join(results_dir, '*.json')))
    before_time = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT,
            capture_output=True, text=True,
            timeout=1800,
            env={**os.environ, 'QMT_LOG_LEVEL': 'WARNING'}
        )

        if result.returncode != 0:
            print(f'  回测进程退出码: {result.returncode}')
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-5:]:
                    print(f'  stderr: {line}')

    except subprocess.TimeoutExpired:
        return {'label': label, 'error': '回测超时(30分钟)'}
    except Exception as e:
        return {'label': label, 'error': f'回测异常: {e}'}

    # 查找新生成的结果文件
    after_files = set(glob.glob(os.path.join(results_dir, '*.json')))
    new_files = after_files - before_files

    if new_files:
        result_path = max(new_files, key=os.path.getmtime)
    else:
        # 没有新文件，找最新的
        all_files = glob.glob(os.path.join(results_dir, '*.json'))
        recent = [f for f in all_files if os.path.getmtime(f) > before_time - 2]
        if recent:
            result_path = max(recent, key=os.path.getmtime)
        else:
            latest = _find_latest_result(STRATEGY_DIR, STRATEGY_NAME, min_age_seconds=0)
            if latest:
                result_path = latest
            else:
                return {'label': label, 'error': '未找到回测结果文件'}

    # 等待文件写入完成
    time.sleep(1)

    metrics = _extract_metrics_from_result(result_path, label, extra_params)

    # 保存到优化结果目录
    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f'  夏普比率: {metrics.get("sharpe_ratio", "N/A")}')
    print(f'  总收益率: {metrics.get("total_return_pct", "N/A")}%')
    print(f'  最大回撤: {metrics.get("max_drawdown_pct", "N/A")}%')

    return metrics


def run_all_optimizations():
    """运行所有单项优化回测"""
    results = []
    baseline = None

    for label, name, params in ALL_OPTIMIZATIONS:
        print(f'\n{"#"*60}')
        print(f'# 优化: {name}')
        print(f'{"#"*60}')

        metrics = run_backtest_with_params(
            extra_params=params, label=label,
            start_date=TRAIN_START, end_date=TRAIN_END
        )

        metrics['name'] = name
        results.append(metrics)

        if label == 'baseline':
            baseline = metrics

    # 输出汇总
    print(f'\n{"="*80}')
    print('优化结果汇总')
    print(f'{"="*80}')

    if baseline:
        baseline_sharpe = baseline.get('sharpe_ratio', 0)
        print(f'\n基线夏普比率: {baseline_sharpe:.4f}')
        print(f'{"─"*80}')
        print(f'{"优化方向":<25} {"夏普":>8} {"变化":>8} {"总收益":>10} {"最大回撤":>10} {"结论":>6}')
        print(f'{"─"*80}')

        for r in results[1:]:
            sharpe = r.get('sharpe_ratio', 0)
            improvement = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
            total_ret = r.get('total_return_pct', 0)
            max_dd = r.get('max_drawdown_pct', 0)
            verdict = '有效' if improvement >= 5 else '无效'
            print(f'{r["name"]:<25} {sharpe:>8.4f} {improvement:>+7.1f}% {total_ret:>9.2f}% {max_dd:>9.2f}% {verdict:>6}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default=None, help='只运行指定优化')
    parser.add_argument('--all', action='store_true', help='运行所有优化')
    parser.add_argument('--oos', action='store_true', help='运行样本外验证')
    parser.add_argument('--params', type=str, default=None, help='额外参数 key=val,key=val')
    args = parser.parse_args()

    if args.all:
        run_all_optimizations()
    elif args.oos:
        # 样本外验证，需要传参
        if args.label and args.params:
            params = {}
            for pair in args.params.split(','):
                k, v = pair.split('=')
                try:
                    if '.' in v:
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v
            print(f'Running OOS validation for {args.label} with params {params}')
            # 样本内
            run_backtest_with_params(extra_params=params, label=f'{args.label}_is',
                                      start_date=TRAIN_START, end_date=TRAIN_END)
            # 样本外
            run_backtest_with_params(extra_params=params, label=f'{args.label}_oos',
                                      start_date=VALID_START, end_date=VALID_END)
            # 基线
            run_backtest_with_params(extra_params=None, label='baseline_is',
                                      start_date=TRAIN_START, end_date=TRAIN_END)
            run_backtest_with_params(extra_params=None, label='baseline_oos',
                                      start_date=VALID_START, end_date=VALID_END)
        else:
            print('OOS validation requires --label and --params')
    elif args.label:
        # 运行指定优化
        for label, name, params in ALL_OPTIMIZATIONS:
            if label == args.label:
                run_backtest_with_params(extra_params=params, label=label,
                                          start_date=TRAIN_START, end_date=TRAIN_END)
                break
        else:
            print(f'Unknown label: {args.label}')
    else:
        # 默认运行所有
        run_all_optimizations()

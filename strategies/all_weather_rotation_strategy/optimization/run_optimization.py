"""全天候轮动策略优化脚本 - 使用命令行回测

通过 subprocess 调用 main.py 运行回测，从结果JSON中提取指标。
避免 BacktestAPI 的 set_ai_mode 导致的异常净值问题。
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

STRATEGY_NAME = 'all_weather_rotation'
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


def _find_latest_result(strategy_dir, strategy_name, min_age_seconds=2):
    """查找最新的回测结果JSON文件"""
    results_dir = os.path.join(strategy_dir, 'backtest_results')
    pattern = os.path.join(results_dir, f'*_{strategy_name}.json')
    files = glob.glob(pattern)
    if not files:
        return None
    # 按修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    for f in files:
        # 确保文件不是正在写入的
        if time.time() - os.path.getmtime(f) > min_age_seconds:
            return f
    # 如果没有足够旧的文件，取最新的
    return files[0]


def _extract_metrics_from_result(result_path, label, extra_params=None):
    """从回测结果JSON中提取关键指标"""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {'label': label, 'error': f'读取结果失败: {e}'}

    metrics = {'label': label, 'extra_params': extra_params}

    # 从 metrics 中提取（标准格式）
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

    # 从 summary 中提取（备选）
    if not m:
        summary = data.get('summary', {})
        if summary:
            for key, val in summary.items():
                if '夏普' in key:
                    metrics['sharpe_ratio'] = float(val) if isinstance(val, (int, float)) else float(str(val).replace('%', '').replace(',', ''))
                elif '最大回撤' in key:
                    metrics['max_drawdown_pct'] = float(str(val).replace('%', '').replace(',', ''))
                elif '年化' in key and '收益' in key:
                    metrics['annual_return_pct'] = float(str(val).replace('%', '').replace(',', ''))
                elif '总收益' in key:
                    metrics['total_return_pct'] = float(str(val).replace('%', '').replace(',', ''))

    # 从 account 中提取（备选）
    if 'final_value' not in metrics or not metrics['final_value']:
        account = data.get('account', {})
        if account:
            metrics.setdefault('final_value', account.get('final_value', account.get('dynamic_rights', 0)))
            metrics.setdefault('total_return_pct', account.get('total_return_pct', account.get('rate', 0) * 100))

    # 直接从顶层提取（备选）
    metrics.setdefault('sharpe_ratio', data.get('sharpe_ratio', 0))
    metrics.setdefault('max_drawdown_pct', data.get('max_drawdown_pct', data.get('max_drawdown', 0)))
    metrics.setdefault('total_return_pct', data.get('total_return_pct', 0))
    metrics.setdefault('annual_return_pct', data.get('annual_return_pct', 0))

    metrics['timestamp'] = datetime.datetime.now().isoformat()
    return metrics


def run_backtest_with_params(extra_params=None, label='test',
                              pool='沪深300', start_date='2020-04-28', end_date='2024-04-28'):
    """通过命令行运行回测并提取结果"""
    # 构建命令行参数
    cmd = [
        sys.executable, MAIN_SCRIPT,
        '--strategy', STRATEGY_NAME,
        '--pool', pool,
        '--start', start_date,
        '--end', end_date,
        '--ai-mode',
    ]

    # 添加策略参数（key=value,key=value 格式）
    if extra_params:
        param_parts = []
        for k, v in extra_params.items():
            if isinstance(v, bool):
                param_parts.append(f'{k}={str(v).lower()}')
            elif isinstance(v, (int, float)):
                param_parts.append(f'{k}={v}')
            else:
                param_parts.append(f'{k}={v}')
        params_str = ','.join(param_parts)
        cmd.extend(['--strategy-params', params_str])

    print(f'\n{"="*60}')
    print(f'Running backtest: {label}')
    print(f'Extra params: {extra_params}')
    print(f'{"="*60}')
    sys.stdout.flush()

    # 记录运行前的结果文件列表
    results_dir = os.path.join(STRATEGY_DIR, 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    before_files = set(glob.glob(os.path.join(results_dir, '*.json')))
    before_time = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT,
            capture_output=True, text=True,
            timeout=1800,  # 30分钟超时
            env={**os.environ, 'QMT_LOG_LEVEL': 'WARNING'}
        )

        if result.returncode != 0:
            print(f'  回测进程退出码: {result.returncode}')
            if result.stderr:
                # 只打印最后几行错误
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-5:]:
                    print(f'  stderr: {line}')

    except subprocess.TimeoutExpired:
        print(f'  回测超时!')
        return {'label': label, 'error': 'timeout'}
    except Exception as e:
        print(f'  回测异常: {e}')
        return {'label': label, 'error': str(e)}

    # 查找新生成的结果文件（等待文件写入完成）
    result_path = None
    for _ in range(10):
        time.sleep(1)
        after_files = set(glob.glob(os.path.join(results_dir, '*.json')))
        new_files = after_files - before_files
        # 也包括在回测期间修改的旧文件
        new_or_modified = [f for f in after_files
                          if f not in before_files or os.path.getmtime(f) >= before_time]
        if new_or_modified:
            result_path = max(new_or_modified, key=os.path.getmtime)
            break

    if not result_path:
        # 回退：取最新的文件
        all_files = list(glob.glob(os.path.join(results_dir, '*.json')))
        if all_files:
            all_files.sort(key=os.path.getmtime, reverse=True)
            result_path = all_files[0]

    if not result_path:
        print(f'  未找到结果文件')
        return {'label': label, 'error': 'no result file'}

    # 提取指标
    metrics = _extract_metrics_from_result(result_path, label, extra_params)

    # 保存到优化结果目录
    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 打印关键指标
    sharpe = metrics.get('sharpe_ratio', 0)
    total_ret = metrics.get('total_return_pct', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)
    print(f'\n--- Results for {label} ---')
    print(f'  Sharpe: {sharpe:.4f}')
    print(f'  Total Return: {total_ret:.2f}%')
    print(f'  Max Drawdown: {max_dd:.2f}%')
    sys.stdout.flush()

    return metrics


def run_out_of_sample_test(extra_params, label):
    """使用固定验证集进行样本外测试"""
    in_sample = run_backtest_with_params(
        extra_params=extra_params,
        label=f'{label}_is', start_date=TRAIN_START, end_date=TRAIN_END)

    out_sample = run_backtest_with_params(
        extra_params=extra_params,
        label=f'{label}_oos', start_date=VALID_START, end_date=VALID_END)

    baseline_is = run_backtest_with_params(
        extra_params=None,
        label='baseline_is', start_date=TRAIN_START, end_date=TRAIN_END)

    baseline_oos = run_backtest_with_params(
        extra_params=None,
        label='baseline_oos', start_date=VALID_START, end_date=VALID_END)

    is_sharpe = in_sample.get('sharpe_ratio', 0)
    oos_sharpe = out_sample.get('sharpe_ratio', 0)
    base_is_sharpe = baseline_is.get('sharpe_ratio', 0)
    base_oos_sharpe = baseline_oos.get('sharpe_ratio', 0)

    is_improvement = (is_sharpe - base_is_sharpe) / abs(base_is_sharpe) * 100 if base_is_sharpe != 0 else 0
    oos_improvement = (oos_sharpe - base_oos_sharpe) / abs(base_oos_sharpe) * 100 if base_oos_sharpe != 0 else 0

    return {
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'valid_period': f'{VALID_START} ~ {VALID_END}',
        'in_sample_sharpe': is_sharpe,
        'out_sample_sharpe': oos_sharpe,
        'baseline_is_sharpe': base_is_sharpe,
        'baseline_oos_sharpe': base_oos_sharpe,
        'is_improvement_pct': is_improvement,
        'oos_improvement_pct': oos_improvement,
        'decay_ratio': oos_improvement / is_improvement if is_improvement != 0 else 0,
    }


def run_parameter_sensitivity_test(param_name, param_value,
                                    label, perturbations=[-0.2, -0.1, 0.1, 0.2]):
    results = {}
    base_result = run_backtest_with_params(
        extra_params={param_name: param_value},
        label=f'{label}_base')

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


def run_temporal_stability_test(extra_params, label, full_start, full_end):
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
            extra_params=extra_params,
            label=f'{label}_{year}', start_date=year_start, end_date=year_end)

        base_result = run_backtest_with_params(
            extra_params=None,
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


# ================================================================
# 优化方案定义
# ================================================================
OPTIMIZATIONS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤 (vol=0.02)', {'max_volatility': 0.02}),
    ('opt02_rebalance_stop_loss', '调仓止损 (5%)', {'rebalance_stop_loss': 0.05}),
    ('opt03_momentum_confirm', '动量确认', {'require_positive_momentum': True}),
    ('opt04_switch_threshold', '换仓阈值 (5%)', {'switch_threshold': 0.05}),
    ('opt05_relaxed_small', '放宽SMALL (ROE>8%,ROA>5%)', {'small_min_roe_relaxed': 0.08, 'small_min_roa_relaxed': 0.05}),
    ('opt06_low_threshold', '降低无敌阈值 (5%)', {'momentum_threshold_low': 5}),
    ('opt07_long_momentum', '20日动量', {'momentum_days_long': 20}),
    ('opt08_max_position', '持仓集中度限制 (15%)', {'max_position_ratio': 0.15}),
    ('opt09_dual_period', '双周期动量验证', {'dual_period_momentum': True}),
    ('opt10_gold_etf', '优先黄金ETF', {'prefer_gold_etf': True}),
]


def run_all_optimizations():
    """运行所有单项优化回测"""
    print("=" * 60)
    print("全天候轮动策略优化 - 单项优化回测")
    print(f"测试集: {TRAIN_START} ~ {TRAIN_END}")
    print("=" * 60)

    results = []
    for label, name, params in OPTIMIZATIONS:
        print(f"\n>>> 运行: {name} ({label})")
        try:
            result = run_backtest_with_params(
                extra_params=params,
                label=label,
                pool='沪深300',
                start_date=TRAIN_START,
                end_date=TRAIN_END)
            sharpe = result.get('sharpe_ratio', 0)
            total_ret = result.get('total_return_pct', 0)
            max_dd = result.get('max_drawdown_pct', 0)
            print(f"    夏普: {sharpe:.4f}, 总收益: {total_ret:.2f}%, 最大回撤: {max_dd:.2f}%")
            results.append({
                'label': label,
                'name': name,
                'params': params,
                'sharpe': sharpe,
                'total_return': total_ret,
                'max_drawdown': max_dd,
                'annual_return': result.get('annual_return_pct', 0),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            results.append({
                'label': label,
                'name': name,
                'params': params,
                'sharpe': 0,
                'error': str(e),
            })

    # 打印汇总
    print("\n" + "=" * 60)
    print("单项优化汇总")
    print("=" * 60)
    baseline_sharpe = results[0]['sharpe'] if results else 0
    for r in results:
        if 'error' in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            improvement = (r['sharpe'] - baseline_sharpe) / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
            verdict = "有效" if improvement >= 5 else "无效"
            print(f"  {r['name']}: 夏普={r['sharpe']:.4f} ({improvement:+.1f}%) 收益={r['total_return']:.2f}% 回撤={r['max_drawdown']:.2f}% [{verdict}]")

    # 保存汇总
    summary_file = os.path.join(RESULTS_DIR, 'optimization_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_sharpe': baseline_sharpe,
            'results': results,
            'timestamp': datetime.datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    return results


def run_combined_optimization(effective_params):
    """运行组合优化回测"""
    print(f"\n>>> 运行组合优化")
    combined_params = {}
    for p in effective_params:
        combined_params.update(p)

    result = run_backtest_with_params(
        extra_params=combined_params,
        label='opt_combined',
        pool='沪深300',
        start_date=TRAIN_START,
        end_date=TRAIN_END)

    sharpe = result.get('sharpe_ratio', 0)
    total_ret = result.get('total_return_pct', 0)
    max_dd = result.get('max_drawdown_pct', 0)
    print(f"    组合优化: 夏普={sharpe:.4f}, 总收益={total_ret:.2f}%, 最大回撤={max_dd:.2f}%")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'single', 'combined', 'oos', 'sensitivity', 'temporal'], default='all')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--params', type=str, default=None)  # JSON string
    args = parser.parse_args()

    if args.mode == 'all':
        run_all_optimizations()
    elif args.mode == 'single':
        if args.label and args.params:
            params = json.loads(args.params)
            result = run_backtest_with_params(extra_params=params, label=args.label)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.mode == 'combined':
        if args.params:
            params_list = json.loads(args.params)
            run_combined_optimization(params_list)
    elif args.mode == 'oos':
        if args.label and args.params:
            params = json.loads(args.params)
            result = run_out_of_sample_test(params, args.label)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.mode == 'sensitivity':
        if args.label and args.params:
            param_info = json.loads(args.params)
            result = run_parameter_sensitivity_test(
                param_info['name'], param_info['value'], args.label)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.mode == 'temporal':
        if args.label and args.params:
            params = json.loads(args.params)
            result = run_temporal_stability_test(
                params, args.label, TRAIN_START, TRAIN_END)
            print(json.dumps(result, indent=2, ensure_ascii=False))

"""银行轮动策略 - 硬逻辑与过度拟合审查

审查内容：
1. switch_threshold 扰动测试（±10%, ±20%）
2. 分年验证（2020-2024每年独立回测）
3. 样本外验证集详细分析
4. 与无滑点结果的对比
"""
import os
import sys
import json
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir

STRATEGY_NAME = 'bank_rotation'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SLIPPAGE = 0.0007
TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'


def run_backtest(extra_params=None, label='test',
                 start_date=TRAIN_START, end_date=TRAIN_END, slippage=SLIPPAGE):
    strategy_class = get_strategy(STRATEGY_NAME)
    default_kwargs = get_strategy_default_kwargs(STRATEGY_NAME)
    backtest_config = get_strategy_backtest_config(STRATEGY_NAME)

    config = dict(backtest_config)
    config['period'] = '1m'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config['slippage'] = slippage
    config.setdefault('benchmark', '000300.SH')

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(**config)
    api.add_strategy(strategy_class, **merged_kwargs)
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
        metrics['slippage'] = slippage
        metrics['timestamp'] = datetime.datetime.now().isoformat()
    else:
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    print('=' * 70)
    print('硬逻辑与过度拟合审查')
    print('=' * 70)

    best_st = 0.005
    best_sp = 0.004
    best_params = {'switch_threshold': best_st, 'spread_threshold': best_sp}

    # ===== 1. 扰动测试 =====
    print('\n[1] switch_threshold 扰动测试...')
    perturbations = {
        '-20%': best_st * 0.8,
        '-10%': best_st * 0.9,
        'base': best_st,
        '+10%': best_st * 1.1,
        '+20%': best_st * 1.2,
    }
    for name, st in perturbations.items():
        r = run_backtest(
            extra_params={'switch_threshold': round(st, 4), 'spread_threshold': best_sp},
            label=f'review_perturb_{name}')
        print(f'  {name:>5} (st={st:.4f}): 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    # ===== 2. 分年验证 =====
    print('\n[2] 分年验证...')
    years = [
        ('2020', '2020-04-28', '2020-12-31'),
        ('2021', '2021-01-01', '2021-12-31'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
        ('2026', '2026-01-01', '2026-04-28'),
    ]
    for year_name, ys, ye in years:
        r = run_backtest(
            extra_params=best_params,
            label=f'review_year_{year_name}',
            start_date=ys, end_date=ye)
        print(f'  {year_name}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    # ===== 3. 样本外验证集详细分析 =====
    print('\n[3] 样本外验证集（2024-04-28 ~ 2026-04-28）...')
    # 不同threshold在验证集的表现
    for st in [0.005, 0.006, 0.007, 0.008, 0.010]:
        r = run_backtest(
            extra_params={'switch_threshold': st, 'spread_threshold': best_sp},
            label=f'review_oos_st{st:.3f}',
            start_date=VALID_START, end_date=VALID_END)
        print(f'  st={st:.3f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    # ===== 4. 无滑点 vs 有滑点对比 =====
    print('\n[4] 无滑点 vs 有滑点对比...')
    for slip_val, slip_name in [(0.0, '无滑点'), (0.0014, '有滑点')]:
        for st in [0.003, 0.005, 0.008]:
            r = run_backtest(
                extra_params={'switch_threshold': st, 'spread_threshold': best_sp},
                label=f'review_slip_{slip_name}_st{st:.3f}',
                slippage=slip_val)
            print(f'  {slip_name} st={st:.3f}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
                  f'收益={r.get("total_return_pct", 0):.2f}%')

    # ===== 5. 逻辑审查：成本占比分析 =====
    print('\n[5] 成本占比分析...')
    total_cost = (SLIPPAGE * 2 + 0.0002 * 2) * 100  # 百分比
    for st in [0.003, 0.005, 0.006, 0.007, 0.008, 0.010]:
        cost_ratio = total_cost / (st * 100) * 100
        net_margin = (st * 100) - total_cost
        print(f'  st={st:.3f}: 换仓差={st*100:.2f}%, 总成本={total_cost:.4f}%, '
              f'成本占比={cost_ratio:.1f}%, 净利润空间={net_margin:.4f}%')

    # ===== 6. 样本外验证集分年 =====
    print('\n[6] 验证集分年（最优参数）...')
    oos_years = [
        ('2024H2', '2024-04-28', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
        ('2026Q1', '2026-01-01', '2026-04-28'),
    ]
    for year_name, ys, ye in oos_years:
        r = run_backtest(
            extra_params=best_params,
            label=f'review_oos_year_{year_name}',
            start_date=ys, end_date=ye)
        print(f'  {year_name}: 夏普={r.get("sharpe_ratio", 0):.4f}, '
              f'收益={r.get("total_return_pct", 0):.2f}%, '
              f'回撤={r.get("max_drawdown_pct", 0):.2f}%')

    print('\n审查完成。')


if __name__ == '__main__':
    main()

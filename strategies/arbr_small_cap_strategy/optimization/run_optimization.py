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
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'arbr_small_cap'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'

# 10项优化方案
OPTIMIZATIONS = [
    ('opt01_volatility_filter', '波动率过滤 (vol=0.03)', {'max_volatility': 0.03}),
    ('opt02_rebalance_stoploss', '调仓止损 (5%)', {'rebalance_stoploss': 0.05}),
    ('opt03_industry_limit', '行业分散 (同行业最多1只)', {'max_same_industry': 1}),
    ('opt04_arbr_period_short', 'ARBR短周期 (14)', {'arbr_period': 14}),
    ('opt05_max_stocks_5', '持仓数量5只', {'max_stocks': 5}),
    ('opt06_arbr_range_narrow', 'ARBR范围收窄 (-0.5, 0.5)', {'arbr_low': -0.5, 'arbr_high': 0.5}),
    ('opt07_turnover_control', '换手率控制 (50%)', {'max_turnover_ratio': 0.5}),
    ('opt08_volume_confirm', '成交量确认 (ratio=0.8)', {'min_volume_ratio': 0.8}),
    ('opt09_biweekly_rebalance', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('opt10_min_market_cap', '市值下限10亿', {'min_market_cap': 10}),
]


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool='中证1000', start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config['pool'] = pool
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


def run_all_optimizations():
    """运行所有优化回测"""
    baseline = run_backtest_with_params(label='baseline')
    baseline_sharpe = baseline.get('sharpe_ratio', 0)
    print(f"\n{'='*60}")
    print(f"基线: Sharpe={baseline_sharpe:.4f}, Return={baseline.get('total_return_pct', 0):.2f}%, "
          f"Drawdown={baseline.get('max_drawdown_pct', 0):.2f}%")
    print(f"{'='*60}\n")

    results = []
    for label, name, params in OPTIMIZATIONS:
        print(f"Running {label}: {name} ...")
        try:
            r = run_backtest_with_params(extra_params=params, label=label)
            sharpe = r.get('sharpe_ratio', 0)
            improvement = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
            verdict = '有效' if improvement >= 5 else '无效'
            print(f"  Sharpe={sharpe:.4f} ({improvement:+.1f}%) Return={r.get('total_return_pct', 0):.2f}% "
                  f"DD={r.get('max_drawdown_pct', 0):.2f}% -> {verdict}")
            results.append({
                'label': label, 'name': name, 'params': params,
                'sharpe': sharpe, 'improvement_pct': improvement, 'verdict': verdict,
                'total_return': r.get('total_return_pct', 0),
                'max_drawdown': r.get('max_drawdown_pct', 0),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'label': label, 'name': name, 'params': params,
                'sharpe': 0, 'improvement_pct': -100, 'verdict': '错误',
                'error': str(e),
            })

    # 汇总
    print(f"\n{'='*60}")
    print(f"优化汇总 (基线Sharpe={baseline_sharpe:.4f})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label']}: Sharpe={r['sharpe']:.4f} ({r['improvement_pct']:+.1f}%) -> {r['verdict']}")

    effective = [r for r in results if r['verdict'] == '有效']
    print(f"\n有效优化: {len(effective)}/{len(results)} 项")
    for r in effective:
        print(f"  {r['label']}: {r['name']} (Sharpe+{r['improvement_pct']:.1f}%)")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default=None, help='Run single optimization by label')
    parser.add_argument('--all', action='store_true', help='Run all optimizations')
    parser.add_argument('--baseline', action='store_true', help='Run baseline only')
    args = parser.parse_args()

    if args.baseline:
        r = run_backtest_with_params(label='baseline')
        print(f"Baseline: Sharpe={r.get('sharpe_ratio', 'N/A'):.4f}, "
              f"Return={r.get('total_return_pct', 'N/A'):.2f}%, "
              f"Drawdown={r.get('max_drawdown_pct', 'N/A'):.2f}%")
    elif args.label:
        # 查找对应优化
        for label, name, params in OPTIMIZATIONS:
            if label == args.label:
                r = run_backtest_with_params(extra_params=params, label=label)
                print(f"{name}: Sharpe={r.get('sharpe_ratio', 0):.4f}, "
                      f"Return={r.get('total_return_pct', 0):.2f}%, "
                      f"Drawdown={r.get('max_drawdown_pct', 0):.2f}%")
                break
    else:
        run_all_optimizations()

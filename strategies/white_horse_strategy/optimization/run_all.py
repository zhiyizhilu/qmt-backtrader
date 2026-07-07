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
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'white_horse'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
POOL = '沪深300'

# 所有优化方案
OPTIMIZATIONS = [
    ('baseline', '基线策略', None),
    ('opt01_volatility_filter', '波动率过滤 (vol=0.03)', {'max_volatility': 0.03}),
    ('opt02_stop_loss', '止损机制 (drawdown=-0.15)', {'stop_loss_pct': -0.15}),
    ('opt03_industry_limit', '行业分散 (max=2)', {'max_same_industry': 2}),
    ('opt04_cold_position', '冷市降仓 (ratio=0.80)', {'cold_position_ratio': 0.80}),
    ('opt05_biweekly', '双周调仓', {'rebalance_freq': 'biweekly'}),
    ('opt06_replace_limit', '换手率限制 (max=3)', {'max_replace': 3}),
    ('opt07_hot_roe_8', '热市ROE提升 (roe=8)', {'hot_min_roe': 8.0}),
    ('opt08_cold_pb_1.5', '冷市放宽PB (pb=1.5)', {'cold_max_pb': 1.5}),
    ('opt09_warm_profit_5', '温市利润增速 (yoy=5)', {'warm_min_profit_yoy': 5}),
    ('opt10_hot_pb_cap', '热市PB上限 (pb=15)', {'hot_max_pb': 15}),
]


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    config['data_lookback_days'] = 400
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
    """逐项运行所有优化回测"""
    results = []
    for label, name, params in OPTIMIZATIONS:
        print(f"\n=== 运行: {name} ({label}) ===")
        if params:
            print(f"    参数: {params}")
        result = run_backtest_with_params(label=label, extra_params=params)
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
        })

    # 输出汇总表
    print("\n\n========== 优化结果汇总 ==========")
    baseline_sharpe = results[0]['sharpe']
    print(f"{'优化':>30s} {'夏普':>8s} {'变化':>8s} {'总收益':>10s} {'回撤':>8s}")
    print("-" * 70)
    for r in results:
        change = (r['sharpe'] - baseline_sharpe) / baseline_sharpe * 100 if baseline_sharpe != 0 else 0
        print(f"{r['name']:>30s} {r['sharpe']:>8.4f} {change:>+7.1f}% {r['total_return']:>9.2f}% {r['max_drawdown']:>7.2f}%")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'single'], default='all')
    parser.add_argument('--label', type=str, default='baseline')
    parser.add_argument('--params', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'all':
        run_all_optimizations()
    else:
        extra_params = json.loads(args.params) if args.params else None
        result = run_backtest_with_params(label=args.label, extra_params=extra_params)
        print(json.dumps(result, indent=2, ensure_ascii=False))

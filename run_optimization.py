import os
import sys
import json
import datetime
import traceback

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from core.stock_selection import StockSelectionStrategy
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config
from core.data.index_constituent import IndexConstituentManager

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backtest_with_params(strategy_name='small_cap', extra_params=None, label='test'):
    print(f'\n{"="*60}')
    print(f'Running backtest: {label}')
    print(f'Extra params: {extra_params}')
    print(f'{"="*60}')
    sys.stdout.flush()

    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = '2020-04-28'
    config['end_date'] = '2026-04-28'
    benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get('中证1000', '000300.SH')
    config.setdefault('benchmark', benchmark)

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.configure(**config)
    api.load_financial_data(sector='中证1000')
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
            metrics['annual_return_pct'] = annual_ret * 100
            metrics['trading_days'] = days
        metrics['label'] = label
        metrics['extra_params'] = extra_params
        metrics['timestamp'] = datetime.datetime.now().isoformat()

        print(f'\n--- Results for {label} ---')
        print(f'Initial: {metrics["initial_capital"]:.2f}')
        print(f'Final: {metrics["final_value"]:.2f}')
        print(f'Total Return: {metrics["total_return_pct"]:.2f}%')
        print(f'Sharpe Ratio: {metrics["sharpe_ratio"]:.4f}')
        print(f'Max Drawdown: {metrics["max_drawdown_pct"]:.2f}%')
        if 'annual_return_pct' in metrics:
            print(f'Annual Return: {metrics["annual_return_pct"]:.2f}%')
            print(f'Trading Days: {metrics["trading_days"]}')
        sys.stdout.flush()
    else:
        print(f'No result for {label}')
        metrics['label'] = label
        metrics['error'] = 'No result'

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'Results saved to {result_file}')
    sys.stdout.flush()

    return metrics


if __name__ == '__main__':
    opt_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    optimizations = {
        0: ('baseline', None),
        1: ('opt01_volatility_filter', {'max_volatility': 0.04}),
        2: ('opt02_volume_confirm', {'min_avg_volume_ratio': 1.0}),
        3: ('opt03_multi_factor_score', {'use_multi_factor_score': True}),
        4: ('opt04_biweekly_rebalance', {'rebalance_freq': 'biweekly'}),
        5: ('opt05_relative_momentum', {'use_relative_momentum': True}),
        6: ('opt06_industry_momentum_weight', {'use_industry_momentum': True}),
        7: ('opt07_stop_loss', {'stop_loss_pct': 0.08}),
        8: ('opt08_min_market_cap', {'min_market_cap': 15}),
        9: ('opt09_financial_quality_score', {'use_quality_score': True}),
        10: ('opt10_turnover_control', {'max_turnover_ratio': 0.5}),
        11: ('opt11_combined_vol_stoploss', {'max_volatility': 0.04, 'stop_loss_pct': 0.08}),
    }

    if opt_num in optimizations:
        label, params = optimizations[opt_num]
        try:
            run_backtest_with_params('small_cap', params, label)
        except Exception as e:
            print(f'Error running {label}: {e}')
            traceback.print_exc()
    else:
        print(f'Unknown optimization number: {opt_num}')
        print(f'Available: {list(optimizations.keys())}')

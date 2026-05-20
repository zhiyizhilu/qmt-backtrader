import os, sys, json, datetime, gc, traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config
from core.data.index_constituent import IndexConstituentManager

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

POOL = '中证全指'
START_DATE = '2020-04-28'
END_DATE = '2026-04-28'

ALL_TESTS = [
    ('baseline', None),
    ('opt01_volatility_filter', {'max_volatility': 0.02}),
    ('opt02_stop_loss', {'stop_loss': -0.05}),
    ('opt03_industry_limit', {'max_industry_stocks': 3}),
    ('opt04_biweekly_rebalance', {'rebalance_freq': 'biweekly'}),
    ('opt05_momentum_confirm', {'min_momentum': 0.02}),
    ('opt06_min_roe', {'min_roe': 0.05}),
    ('opt07_max_stocks_15', {'max_stocks': 15}),
    ('opt08_ic_abs_weight', {'use_ic_abs_weight': True}),
    ('opt09_quality_score', {'min_quality_score': 2}),
    ('opt10_combined_vol_stoploss', {'max_volatility': 0.02, 'stop_loss': -0.05}),
]

force = '--force' in sys.argv or '-f' in sys.argv
start_idx = 0
end_idx = len(ALL_TESTS)
args = [a for a in sys.argv[1:] if not a.startswith('-')]
if len(args) >= 1:
    start_idx = int(args[0])
if len(args) >= 2:
    end_idx = int(args[1])

def extract_metrics(result, label, extra_params):
    metrics = {'label': label, 'extra_params': extra_params}
    if result is None:
        metrics['error'] = 'No result object'
        return metrics
    try:
        metrics['sharpe_ratio'] = result.sharpe_ratio()
    except Exception as e:
        metrics['sharpe_ratio'] = None
        metrics['sharpe_error'] = str(e)
    try:
        metrics['max_drawdown_pct'] = result.max_drawdown() * 100
    except Exception as e:
        metrics['max_drawdown_pct'] = None
        metrics['dd_error'] = str(e)
    try:
        acc = result.account
        if acc is not None:
            metrics['total_return_pct'] = acc.rate * 100
        else:
            metrics['total_return_pct'] = None
            metrics['acc_error'] = 'account is None'
    except Exception as e:
        metrics['total_return_pct'] = None
        metrics['acc_error'] = str(e)
    try:
        if result.df is not None and len(result.df) > 0:
            days = len(result.df)
            years = days / 252
            rate = metrics.get('total_return_pct')
            if rate is not None and years > 0:
                annual_ret = (1 + rate / 100) ** (1 / years) - 1
                if isinstance(annual_ret, complex):
                    annual_ret = annual_ret.real
                metrics['annual_return_pct'] = float(annual_ret) * 100
                metrics['trading_days'] = days
    except Exception as e:
        metrics['annual_error'] = str(e)
    metrics['timestamp'] = datetime.datetime.now().isoformat()
    return metrics

for i in range(start_idx, min(end_idx, len(ALL_TESTS))):
    label, extra_params = ALL_TESTS[i]
    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    if os.path.exists(result_file) and not force:
        print(f'[{label}] Already exists, skipping (use --force to overwrite)')
        continue

    print(f'\n=== [{i+1}/{len(ALL_TESTS)}] {label} ===')
    print(f'    extra_params: {extra_params}')
    api = None
    try:
        strategy_class = get_strategy('medical_multi_factor')
        default_kwargs = get_strategy_default_kwargs('medical_multi_factor')
        backtest_config = get_strategy_backtest_config('medical_multi_factor')

        config = dict(backtest_config)
        config['period'] = '1d'
        config['start_date'] = START_DATE
        config['end_date'] = END_DATE
        benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(POOL, '000985.SH')
        config.setdefault('benchmark', benchmark)

        merged_kwargs = dict(default_kwargs)
        if extra_params:
            merged_kwargs.update(extra_params)

        print(f'    merged_kwargs: {merged_kwargs}')

        api = BacktestAPI()
        api.set_ai_mode(True)
        api.set_no_record(True)
        api.configure(**config)
        api.load_financial_data(sector=POOL)
        api.add_stock_selection_strategy(strategy_class, **merged_kwargs)
        engine_result = api.run()

        bt_result = api.get_result()
        metrics = extract_metrics(bt_result, label, extra_params)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        sr_val = metrics.get('sharpe_ratio', 'N/A')
        ret_val = metrics.get('total_return_pct', 'N/A')
        dd_val = metrics.get('max_drawdown_pct', 'N/A')
        print(f'[{label}] SR={sr_val} | Ret={ret_val} | DD={dd_val}')

    except Exception as e:
        tb = traceback.format_exc()
        print(f'[{label}] ERROR: {e}')
        print(tb)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({'label': label, 'error': str(e), 'traceback': tb}, f, indent=2, ensure_ascii=False)

    finally:
        try:
            if api:
                del api
        except Exception:
            pass
        gc.collect()

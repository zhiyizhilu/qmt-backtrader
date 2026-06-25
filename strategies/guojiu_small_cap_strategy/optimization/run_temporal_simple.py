"""简化版时间稳定性测试 - 逐年份运行"""
import os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'guojiu_small_cap'
RESULTS_DIR = os.path.join(get_strategy_dir(STRATEGY_NAME), 'optimization', 'optimization_results')
POOL = '中小综指'
STOCKS8 = {'max_stocks': 8, 'ma_stock_nums': (5, 6, 8, 9, 10)}

def run(extra_params, label, ys, ye):
    sc = get_strategy(STRATEGY_NAME)
    dk = get_strategy_default_kwargs(STRATEGY_NAME)
    bc = get_strategy_backtest_config(STRATEGY_NAME)
    c = dict(bc); c['period'] = '1d'; c['start_date'] = ys; c['end_date'] = ye
    c.setdefault('benchmark', IndexConstituentManager.SECTOR_TO_INDEX.get(POOL, '000300.SH'))
    m = dict(dk)
    if extra_params: m.update(extra_params)
    api = BacktestAPI(); api.set_ai_mode(True); api.set_no_record(True)
    api.configure(**c); api.load_financial_data(sector=POOL)
    api.add_stock_selection_strategy(sc, **m); api.run()
    r = api.get_result()
    d = {}
    if r:
        d['sharpe_ratio'] = r.sharpe_ratio()
        d['total_return_pct'] = r.account.rate * 100
        d['max_drawdown_pct'] = r.max_drawdown() * 100
    d['label'] = label
    with open(os.path.join(RESULTS_DIR, f'{label}.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return d

if __name__ == '__main__':
    years = [(2020, '2020-04-28', '2020-12-31'),
             (2021, '2021-01-01', '2021-12-31'),
             (2022, '2022-01-01', '2022-12-31'),
             (2023, '2023-01-01', '2023-12-31'),
             (2024, '2024-01-01', '2024-04-28')]
    results = []
    for year, ys, ye in years:
        opt_label = f'r2_temporal_opt_{year}'
        base_label = f'r2_temporal_base_{year}'
        # Skip if already exists
        opt_path = os.path.join(RESULTS_DIR, f'{opt_label}.json')
        base_path = os.path.join(RESULTS_DIR, f'{base_label}.json')
        if not os.path.exists(opt_path):
            print(f'Running opt {year}...')
            o = run(STOCKS8, opt_label, ys, ye)
        else:
            o = json.load(open(opt_path, 'r', encoding='utf-8'))
        if not os.path.exists(base_path):
            print(f'Running base {year}...')
            b = run(None, base_label, ys, ye)
        else:
            b = json.load(open(base_path, 'r', encoding='utf-8'))
        os_val = o.get('sharpe_ratio', 0)
        bs_val = b.get('sharpe_ratio', 0)
        imp = os_val - bs_val
        pos = imp > 0
        results.append((year, os_val, bs_val, imp, pos))
        print(f'{year}: opt={os_val:.4f} base={bs_val:.4f} diff={imp:+.4f} {"POS" if pos else "NEG"}')

    positive = sum(1 for r in results if r[4])
    total = len(results)
    ratio = positive / total if total else 0
    print(f'\nPositive: {positive}/{total}, Ratio: {ratio:.2f}')

import os
import sys
import json
import datetime
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'
os.environ['QMT_CACHE_DIR'] = os.path.join(PROJECT_ROOT, '.cache')

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config, get_strategy_dir
from core.data.index_constituent import IndexConstituentManager

STRATEGY_NAME = 'dividend_value_growth'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START = '2020-04-28'
TRAIN_END = '2024-04-28'
VALID_START = '2024-04-28'
VALID_END = '2026-04-28'
POOL = '中证1000'


def run_backtest_with_params(strategy_name=STRATEGY_NAME, extra_params=None, label='test',
                              pool=POOL, start_date=TRAIN_START, end_date=TRAIN_END):
    strategy_class = get_strategy(strategy_name)
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)

    config = dict(backtest_config)
    config['period'] = '1d'
    config['start_date'] = start_date
    config['end_date'] = end_date
    benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(pool, '000300.SH')
    config.setdefault('benchmark', benchmark)

    merged_kwargs = dict(default_kwargs)
    if extra_params:
        merged_kwargs.update(extra_params)

    try:
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
        else:
            metrics['error'] = 'No result'
    except Exception as e:
        metrics = {'error': str(e), 'traceback': traceback.format_exc()}

    metrics['label'] = label
    metrics['extra_params'] = extra_params
    metrics['timestamp'] = datetime.datetime.now().isoformat()

    result_file = os.path.join(RESULTS_DIR, f'{label}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    sharpe = metrics.get('sharpe_ratio', 0)
    ret = metrics.get('total_return_pct', 0)
    dd = metrics.get('max_drawdown_pct', 0)
    print(f"  {label}: Sharpe={sharpe:.4f}, Return={ret:.2f}%, DD={dd:.2f}%")
    return metrics


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'oos':
        # 样本外验证
        print("=== 样本外验证 (opt02 止损10%) ===")
        # 测试集基线
        print("Running baseline IS...")
        baseline_is = run_backtest_with_params(label='review_baseline_is', start_date=TRAIN_START, end_date=TRAIN_END)
        # 验证集基线
        print("Running baseline OOS...")
        baseline_oos = run_backtest_with_params(label='review_baseline_oos', start_date=VALID_START, end_date=VALID_END)
        # 测试集优化
        print("Running opt02 IS...")
        opt_is = run_backtest_with_params(extra_params={'stop_loss_pct': 0.10}, label='review_opt02_is', start_date=TRAIN_START, end_date=TRAIN_END)
        # 验证集优化
        print("Running opt02 OOS...")
        opt_oos = run_backtest_with_params(extra_params={'stop_loss_pct': 0.10}, label='review_opt02_oos', start_date=VALID_START, end_date=VALID_END)

        is_imp = (opt_is.get('sharpe_ratio', 0) - baseline_is.get('sharpe_ratio', 0)) / abs(baseline_is.get('sharpe_ratio', 1)) * 100
        oos_imp = (opt_oos.get('sharpe_ratio', 0) - baseline_oos.get('sharpe_ratio', 0)) / abs(baseline_oos.get('sharpe_ratio', 1)) * 100
        decay = oos_imp / is_imp if is_imp != 0 else 0

        print(f"\n=== 样本外验证结果 ===")
        print(f"测试集: 基线夏普={baseline_is.get('sharpe_ratio',0):.4f}, 优化夏普={opt_is.get('sharpe_ratio',0):.4f}, 提升={is_imp:+.1f}%")
        print(f"验证集: 基线夏普={baseline_oos.get('sharpe_ratio',0):.4f}, 优化夏普={opt_oos.get('sharpe_ratio',0):.4f}, 提升={oos_imp:+.1f}%")
        print(f"衰减比: {decay:.2f}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'sensitivity':
        # 参数敏感性分析
        print("=== 参数敏感性分析 (止损阈值) ===")
        base_value = 0.10
        perturbations = [0.05, 0.08, 0.12, 0.15]
        for val in perturbations:
            print(f"Testing stop_loss_pct={val}...")
            run_backtest_with_params(extra_params={'stop_loss_pct': val}, label=f'sensitivity_sl_{int(val*100)}')

    elif len(sys.argv) > 1 and sys.argv[1] == 'temporal':
        # 时间分段稳定性
        print("=== 时间分段稳定性 ===")
        years = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
        for y_start, y_end in years:
            sd = f'{y_start}-04-28'
            ed = f'{y_end}-04-28'
            print(f"\n--- {y_start}-{y_end} ---")
            print("Baseline:")
            run_backtest_with_params(label=f'temporal_baseline_{y_start}', start_date=sd, end_date=ed)
            print("Opt02:")
            run_backtest_with_params(extra_params={'stop_loss_pct': 0.10}, label=f'temporal_opt02_{y_start}', start_date=sd, end_date=ed)

    elif len(sys.argv) > 1 and sys.argv[1] == 'combined':
        # 组合优化：opt02止损 + opt01波动率
        print("=== 组合优化 ===")
        run_backtest_with_params(
            extra_params={'stop_loss_pct': 0.10, 'max_volatility': 0.03},
            label='combined_stoploss_vol')

    else:
        print("Usage: python review.py [oos|sensitivity|temporal|combined]")

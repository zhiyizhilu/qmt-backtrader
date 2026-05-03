import json
import os
import datetime
import logging
from typing import Dict, List, Optional, Any

import pandas as pd

from core.models import BacktestingResult, TradeRecord


_BACKTEST_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backtest_results'
)
_INDEX_FILE = 'index.json'
_FRAMEWORK_VERSION = '1.0'


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _run_id(strategy_name: str, dt: Optional[datetime.datetime] = None) -> str:
    if dt is None:
        dt = datetime.datetime.now()
    ts = dt.strftime('%Y%m%d_%H%M%S')
    return f'{ts}_{strategy_name}'


def _serialize_trade_record(tr: TradeRecord) -> Dict[str, Any]:
    d = {
        'order_id': tr.order_id,
        'instrument_id': tr.instrument_id,
        'direction': tr.direction,
        'offset': tr.offset,
        'volume': tr.volume,
        'order_price': tr.order_price,
        'trade_price': tr.trade_price,
        'fee': tr.fee,
        'pnl': tr.pnl,
        'memo': tr.memo,
    }
    if tr.trade_time is not None:
        if isinstance(tr.trade_time, (datetime.datetime, pd.Timestamp)):
            d['trade_time'] = tr.trade_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            d['trade_time'] = str(tr.trade_time)
    else:
        d['trade_time'] = None
    return d


def _serialize_equity_curve(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        dt_val = row.get('datetime')
        if isinstance(dt_val, (datetime.datetime, pd.Timestamp)):
            date_str = dt_val.strftime('%Y-%m-%d')
        else:
            date_str = str(dt_val)[:10]
        records.append({
            'date': date_str,
            'portfolio_value': float(row.get('PortfolioValue', 0)),
            'pnl': float(row.get('PnL', 0)),
        })
    return records


def _serialize_benchmark_curve(benchmark_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if benchmark_df is None or benchmark_df.empty:
        return []
    records = []
    for idx, row in benchmark_df.iterrows():
        if isinstance(idx, (datetime.datetime, pd.Timestamp)):
            date_str = idx.strftime('%Y-%m-%d')
        else:
            date_str = str(idx)[:10]
        records.append({
            'date': date_str,
            'close': float(row.get('close', 0)),
        })
    return records


class BacktestRecorder:
    """回测结果记录与管理器

    功能：
    - 自动记录每次回测的完整结果到本地 JSON 文件
    - 维护全局索引便于检索
    - 支持加载、列举、对比历史回测数据
    - 生成 HTML + Plotly 可视化对比报告
    """

    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = results_dir or _BACKTEST_RESULTS_DIR
        self.index_path = os.path.join(self.results_dir, _INDEX_FILE)
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        _ensure_dir(self.results_dir)

    def record(
        self,
        result: BacktestingResult,
        strategy_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录单次回测结果，返回 run_id"""
        now = datetime.datetime.now()
        run_id = _run_id(strategy_name, now)

        result.prepare_data()

        acc = result.account
        metrics: Dict[str, Any] = {
            'initial_capital': acc.initial_capital,
            'final_value': acc.dynamic_rights,
            'total_profit': acc.total_profit,
            'total_return_pct': round(acc.rate * 100, 4),
            'fee': round(acc.fee, 2),
            'sharpe_ratio': round(result.sharpe_ratio(), 4),
            'max_drawdown_pct': round(result.max_drawdown() * 100, 4),
            'annual_return_pct': round(result.annual_return(acc.initial_capital, acc.dynamic_rights) * 100, 4),
            'total_trading_days': result.total_trading_days,
            'win_days': result.pnl_days[0],
            'loss_days': result.pnl_days[1],
            'turnover': round(result.turnover, 2),
            'total_volume': result.total_volume,
        }

        trade_log = [_serialize_trade_record(tr) for tr in result.trade_log]
        equity_curve = _serialize_equity_curve(result.df)
        benchmark_curve = _serialize_benchmark_curve(result.benchmark_df)

        data = {
            'meta': {
                'run_id': run_id,
                'strategy_name': strategy_name,
                'timestamp': now.isoformat(),
                'framework_version': _FRAMEWORK_VERSION,
            },
            'config': config or {},
            'strategy_params': _make_json_safe(result.strategy_params),
            'metrics': metrics,
            'trade_log': trade_log,
            'equity_curve': equity_curve,
            'benchmark_curve': benchmark_curve,
        }

        strategy_dir = os.path.join(self.results_dir, strategy_name)
        _ensure_dir(strategy_dir)

        file_path = os.path.join(strategy_dir, f'{run_id}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        index_entry = {
            'run_id': run_id,
            'strategy_name': strategy_name,
            'timestamp': now.isoformat(),
            'total_return_pct': metrics['total_return_pct'],
            'annual_return_pct': metrics['annual_return_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'total_trading_days': metrics['total_trading_days'],
            'file': f'{strategy_name}/{run_id}.json',
        }
        self._update_index(index_entry)

        self.logger.info(f'[BacktestRecorder] 回测结果已记录: {run_id} -> {file_path}')
        return run_id

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        """加载指定回测的完整数据"""
        index = self._load_index()
        for entry in index:
            if entry['run_id'] == run_id:
                file_path = os.path.join(self.results_dir, entry['file'])
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    self.logger.warning(f'[BacktestRecorder] 文件不存在: {file_path}')
                break
        return None

    def list_records(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出回测记录摘要，支持按策略名过滤"""
        index = self._load_index()
        if strategy_name:
            return [e for e in index if e['strategy_name'] == strategy_name]
        return index

    def compare(self, run_ids: List[str]) -> Dict[str, Any]:
        """多组回测结果对比分析"""
        records = []
        for rid in run_ids:
            data = self.load(rid)
            if data:
                records.append(data)
            else:
                self.logger.warning(f'[BacktestRecorder] 未找到回测记录: {rid}')

        if not records:
            return {'records': [], 'ranking': []}

        ranking = sorted(records, key=lambda r: r['metrics']['sharpe_ratio'], reverse=True)

        return {
            'records': records,
            'ranking': [
                {
                    'run_id': r['meta']['run_id'],
                    'strategy_name': r['meta']['strategy_name'],
                    'sharpe_ratio': r['metrics']['sharpe_ratio'],
                    'total_return_pct': r['metrics']['total_return_pct'],
                    'max_drawdown_pct': r['metrics']['max_drawdown_pct'],
                    'annual_return_pct': r['metrics']['annual_return_pct'],
                }
                for r in ranking
            ],
        }

    def generate_report(
        self,
        run_ids: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        """生成 HTML + Plotly 可视化对比报告"""
        from utils.report_generator import generate_html_report

        records = []
        for rid in run_ids:
            data = self.load(rid)
            if data:
                records.append(data)

        if not records:
            self.logger.warning('[BacktestRecorder] 无有效回测记录，无法生成报告')
            return ''

        if output_path is None:
            ts = datetime.datetime.now().strftime('%Y%m%d')
            comp_dir = os.path.join(self.results_dir, 'comparison')
            _ensure_dir(comp_dir)
            if len(records) == 1:
                name = records[0]['meta']['strategy_name']
                output_path = os.path.join(comp_dir, f'{ts}_{name}_report.html')
            else:
                output_path = os.path.join(comp_dir, f'{ts}_comp_{len(records)}runs.html')

        return generate_html_report(records, output_path)

    def _load_index(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _update_index(self, entry: Dict[str, Any]):
        index = self._load_index()
        index.append(entry)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (datetime.datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, set):
        return list(obj)
    return str(obj)

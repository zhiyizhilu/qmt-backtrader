import os
import logging
from typing import Dict, List, Any

from utils.plotly_templates import (
    build_equity_curve_figure,
    build_drawdown_figure,
    build_metrics_bar_figure,
    build_trade_stats_table,
    build_summary_cards_html,
    wrap_html,
)

_PLOTLY_JS_FILENAME = 'plotly.min.js'


def _ensure_plotly_js(results_dir: str) -> str:
    """确保 plotly.min.js 存在于 results_dir 下，返回相对于报告文件的路径"""
    js_path = os.path.join(results_dir, _PLOTLY_JS_FILENAME)
    if not os.path.exists(js_path):
        import plotly
        js_content = plotly.offline.get_plotlyjs()
        os.makedirs(results_dir, exist_ok=True)
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_content)
        logging.getLogger(__name__).info(f'[report_generator] Plotly JS 已保存: {js_path}')
    return os.path.relpath(js_path, os.path.dirname(results_dir))


def generate_html_report(records: List[Dict[str, Any]], output_path: str, results_dir: str = '') -> str:
    """根据回测记录列表生成 HTML 可视化报告

    Args:
        records: 回测记录列表，每项为 BacktestRecorder.record() 生成的完整 JSON 数据
        output_path: 输出 HTML 文件路径
        results_dir: backtest_results 根目录，用于存放 plotly.min.js

    Returns:
        生成的 HTML 文件路径
    """
    logger = logging.getLogger(__name__)

    try:
        import plotly
    except ImportError:
        logger.error('[report_generator] plotly 未安装，无法生成 HTML 报告。请运行: pip install plotly')
        return ''

    if not results_dir:
        results_dir = os.path.dirname(os.path.dirname(output_path))

    _ensure_plotly_js(results_dir)

    report_dir = os.path.dirname(output_path)
    js_rel_path = os.path.relpath(os.path.join(results_dir, _PLOTLY_JS_FILENAME), report_dir).replace('\\', '/')

    cards_html = build_summary_cards_html(records)
    equity_fig = build_equity_curve_figure(records)
    drawdown_fig = build_drawdown_figure(records)
    metrics_fig = build_metrics_bar_figure(records)
    trade_table_html = build_trade_stats_table(records)

    html = wrap_html(
        cards_html=cards_html,
        equity_fig=equity_fig,
        drawdown_fig=drawdown_fig,
        metrics_fig=metrics_fig,
        trade_table_html=trade_table_html,
        title=_build_title(records),
        plotly_js_src=js_rel_path,
    )

    os.makedirs(report_dir or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f'[report_generator] 报告已生成: {output_path}')
    return output_path


def _build_title(records: List[Dict[str, Any]]) -> str:
    if len(records) == 1:
        name = records[0]['meta']['strategy_name']
        ts = records[0]['meta']['timestamp'][:10]
        return f'回测报告 — {name} ({ts})'
    names = sorted(set(r['meta']['strategy_name'] for r in records))
    return f'回测对比报告 — {", ".join(names)} ({len(records)}组)'

from typing import Dict, List, Any


_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
]


def _get_color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


def _label(record: Dict[str, Any]) -> str:
    meta = record['meta']
    ts = meta['timestamp'][:16].replace('T', ' ')
    return f"{meta['strategy_name']} ({ts})"


def build_summary_cards_html(records: List[Dict[str, Any]]) -> str:
    cards = []
    for i, rec in enumerate(records):
        m = rec['metrics']
        color = _get_color(i)
        lbl = _label(rec)
        cards.append(f'''
        <div class="card" style="border-left: 4px solid {color};">
            <div class="card-title">{lbl}</div>
            <div class="card-grid">
                <div class="metric"><span class="metric-label">总收益率</span><span class="metric-value {"positive" if m["total_return_pct"] >= 0 else "negative"}">{m["total_return_pct"]:.2f}%</span></div>
                <div class="metric"><span class="metric-label">年化收益</span><span class="metric-value {"positive" if m["annual_return_pct"] >= 0 else "negative"}">{m["annual_return_pct"]:.2f}%</span></div>
                <div class="metric"><span class="metric-label">夏普比率</span><span class="metric-value">{m["sharpe_ratio"]:.4f}</span></div>
                <div class="metric"><span class="metric-label">最大回撤</span><span class="metric-value negative">{m["max_drawdown_pct"]:.2f}%</span></div>
                <div class="metric"><span class="metric-label">交易天数</span><span class="metric-value">{m["total_trading_days"]}</span></div>
                <div class="metric"><span class="metric-label">盈利天数</span><span class="metric-value positive">{m["win_days"]}</span></div>
                <div class="metric"><span class="metric-label">亏损天数</span><span class="metric-value negative">{m["loss_days"]}</span></div>
                <div class="metric"><span class="metric-label">手续费</span><span class="metric-value">{m["fee"]:.2f}</span></div>
            </div>
        </div>''')
    return '\n'.join(cards)


def build_equity_curve_figure(records: List[Dict[str, Any]]) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, rec in enumerate(records):
        curve = rec.get('equity_curve', [])
        if not curve:
            continue
        dates = [p['date'] for p in curve]
        values = [p['portfolio_value'] for p in curve]
        initial = values[0] if values else 1
        normalized = [v / initial for v in values]
        fig.add_trace(go.Scatter(
            x=dates, y=normalized,
            name=_label(rec),
            line=dict(color=_get_color(i), width=2),
            hovertemplate='%{x}<br>净值: %{y:.4f}<extra></extra>',
        ))

        bcurve = rec.get('benchmark_curve', [])
        if bcurve:
            b_dates = [p['date'] for p in bcurve]
            b_closes = [p['close'] for p in bcurve]
            b_initial = b_closes[0] if b_closes else 1
            b_normalized = [c / b_initial for c in b_closes]
            fig.add_trace(go.Scatter(
                x=b_dates, y=b_normalized,
                name=f'{_label(rec)} 基准',
                line=dict(color=_get_color(i), width=1, dash='dash'),
                hovertemplate='%{x}<br>基准净值: %{y:.4f}<extra></extra>',
            ))

    fig.update_layout(
        title='资产净值曲线对比（归一化）',
        xaxis_title='日期',
        yaxis_title='净值',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_drawdown_figure(records: List[Dict[str, Any]]) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, rec in enumerate(records):
        curve = rec.get('equity_curve', [])
        if not curve:
            continue
        values = [p['portfolio_value'] for p in curve]
        dates = [p['date'] for p in curve]
        peak = values[0]
        drawdowns = []
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        fig.add_trace(go.Scatter(
            x=dates, y=drawdowns,
            name=_label(rec),
            fill='tozeroy',
            line=dict(color=_get_color(i), width=1.5),
            hovertemplate='%{x}<br>回撤: %{y:.2f}%<extra></extra>',
        ))

    fig.update_layout(
        title='回撤曲线对比',
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_metrics_bar_figure(records: List[Dict[str, Any]]) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = [_label(rec) for rec in records]
    sharpe = [rec['metrics']['sharpe_ratio'] for rec in records]
    returns = [rec['metrics']['total_return_pct'] for rec in records]
    drawdown = [abs(rec['metrics']['max_drawdown_pct']) for rec in records]
    annual = [rec['metrics']['annual_return_pct'] for rec in records]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('夏普比率', '总收益率 (%)', '最大回撤 (% 绝对值)', '年化收益率 (%)'),
    )

    fig.add_trace(go.Bar(x=labels, y=sharpe, marker_color=[_get_color(i) for i in range(len(records))]), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=returns, marker_color=[_get_color(i) for i in range(len(records))]), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=drawdown, marker_color=[_get_color(i) for i in range(len(records))]), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=annual, marker_color=[_get_color(i) for i in range(len(records))]), row=2, col=2)

    fig.update_layout(
        title='关键指标对比',
        showlegend=False,
        template='plotly_white',
        height=700,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_trade_stats_table(records: List[Dict[str, Any]]) -> str:
    rows = []
    for i, rec in enumerate(records):
        m = rec['metrics']
        trades = rec.get('trade_log', [])
        buy_trades = [t for t in trades if t['direction'] == '0']
        sell_trades = [t for t in trades if t['direction'] == '1']
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_trades = [t for t in trades if t.get('pnl', 0) > 0]
        loss_trades = [t for t in trades if t.get('pnl', 0) < 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        avg_win = sum(t['pnl'] for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(t['pnl'] for t in loss_trades) / len(loss_trades) if loss_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in win_trades) / sum(t['pnl'] for t in loss_trades)) if loss_trades and sum(t['pnl'] for t in loss_trades) != 0 else float('inf')

        rows.append(f'''
        <tr>
            <td style="border-left: 4px solid {_get_color(i)}; font-weight: bold;">{_label(rec)}</td>
            <td>{len(trades)}</td>
            <td>{len(buy_trades)}</td>
            <td>{len(sell_trades)}</td>
            <td class="{"positive" if total_pnl >= 0 else "negative"}">{total_pnl:.2f}</td>
            <td>{win_rate:.1f}%</td>
            <td class="positive">{avg_win:.2f}</td>
            <td class="negative">{avg_loss:.2f}</td>
            <td>{profit_factor:.2f}</td>
        </tr>''')

    return f'''
    <table class="stats-table">
        <thead>
            <tr>
                <th>策略</th><th>总交易数</th><th>买入</th><th>卖出</th>
                <th>总盈亏</th><th>胜率</th><th>平均盈利</th><th>平均亏损</th><th>盈亏比</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>'''


def wrap_html(
    cards_html: str,
    equity_fig: str,
    drawdown_fig: str,
    metrics_fig: str,
    trade_table_html: str,
    title: str,
) -> str:
    import plotly

    plotly_js = f'<script type="text/javascript">{plotly.offline.get_plotlyjs()}</script>'

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {plotly_js}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            padding: 24px;
        }}
        h1 {{
            text-align: center;
            font-size: 24px;
            margin-bottom: 24px;
            color: #1a1a2e;
        }}
        .cards-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            flex: 1;
            min-width: 300px;
            background: #fff;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card-title {{
            font-size: 14px;
            font-weight: 600;
            color: #555;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}
        .metric {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 12px;
            color: #888;
        }}
        .metric-value {{
            font-size: 16px;
            font-weight: 700;
        }}
        .metric-value.positive {{ color: #27ae60; }}
        .metric-value.negative {{ color: #e74c3c; }}
        .section {{
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .stats-table th, .stats-table td {{
            padding: 10px 12px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        .stats-table td.positive {{ color: #27ae60; font-weight: 600; }}
        .stats-table td.negative {{ color: #e74c3c; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="cards-container">
        {cards_html}
    </div>

    <div class="section">
        {equity_fig}
    </div>

    <div class="section">
        {drawdown_fig}
    </div>

    <div class="section">
        {metrics_fig}
    </div>

    <div class="section">
        <h3 style="margin-bottom: 12px;">交易统计</h3>
        {trade_table_html}
    </div>
</body>
</html>'''

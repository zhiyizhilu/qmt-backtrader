import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def get_latest_backtest_result(strategy_dir):
    results_dir = os.path.join(strategy_dir, 'backtest_results')
    if not os.path.isdir(results_dir):
        return None
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    if not json_files:
        return None
    latest_file = json_files[-1]
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _heatmap_color(value):
    if value is None:
        return 'rgba(148, 163, 184, 0.05)'
    if value > 5:
        return 'rgba(52, 211, 153, 0.7)'
    if value > 2:
        return 'rgba(52, 211, 153, 0.4)'
    if value > 0:
        return 'rgba(52, 211, 153, 0.2)'
    if value > -2:
        return 'rgba(248, 113, 113, 0.2)'
    if value > -5:
        return 'rgba(248, 113, 113, 0.4)'
    return 'rgba(248, 113, 113, 0.7)'


def _heatmap_text_color(value):
    if value is None:
        return 'transparent'
    if abs(value) > 2:
        return '#fff'
    return '#94a3b8'


def generate_report():
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    data = get_latest_backtest_result(strategy_dir)

    if not data:
        print(f"未找到回测结果，跳过报告生成")
        return

    meta = data.get('meta', {})
    config = data.get('config', {})
    metrics = data.get('metrics', {})
    trade_log = data.get('trade_log', [])
    equity_curve = data.get('equity_curve', [])
    strategy_params = data.get('strategy_params', {})

    strategy_name = meta.get('strategy_name', 'unknown')
    strategy_cn_name = '霍华·罗斯曼审慎致富价值精选策略'
    start_date = config.get('start_date', '')
    end_date = config.get('end_date', '')
    pool = config.get('pool', '')
    benchmark = config.get('benchmark', '000300.SH')
    commission = config.get('commission', 0)

    initial_capital = metrics.get('initial_capital', 0)
    final_value = metrics.get('final_value', 0)
    total_return_pct = metrics.get('total_return_pct', 0)
    annual_return_pct = metrics.get('annual_return_pct', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    max_drawdown_pct = metrics.get('max_drawdown_pct', 0)
    total_trading_days = metrics.get('total_trading_days', 0)
    win_days = metrics.get('win_days', 0)
    loss_days = metrics.get('loss_days', 0)
    win_rate_pct = round(win_days / (win_days + loss_days) * 100, 1) if (win_days + loss_days) > 0 else 0
    turnover = metrics.get('turnover', 0)
    fee = metrics.get('fee', 0)
    total_profit = metrics.get('total_profit', 0)

    monthly_returns = {}
    if equity_curve:
        monthly_values = {}
        for item in equity_curve:
            ym = item['date'][:7]
            monthly_values[ym] = item['portfolio_value']
        sorted_months = sorted(monthly_values.keys())
        for i in range(1, len(sorted_months)):
            prev_val = monthly_values[sorted_months[i - 1]]
            curr_val = monthly_values[sorted_months[i]]
            if prev_val > 0:
                ret = (curr_val - prev_val) / prev_val * 100
                monthly_returns[sorted_months[i]] = round(ret, 2)

    heatmap_years = sorted(set(ym[:4] for ym in monthly_returns.keys()))
    heatmap_months_labels = [f'{i}月' for i in range(1, 13)]
    heatmap_data = []
    for year in heatmap_years:
        row = []
        for month in range(1, 13):
            ym = f'{year}-{month:02d}'
            row.append(monthly_returns.get(ym, None))
        heatmap_data.append(row)

    recent_trades = trade_log[-50:] if len(trade_log) > 50 else trade_log
    trade_rows = ""
    for t in recent_trades:
        direction = '买入' if t.get('direction') == '0' else '卖出'
        dir_class = 'trade-buy' if direction == '买入' else 'trade-sell'
        pnl_val = t.get('pnl', 0)
        pnl_class = 'up' if pnl_val > 0 else ('down' if pnl_val < 0 else '')
        pnl_str = f'{pnl_val:+.2f}' if pnl_val != 0 else '-'
        trade_time = t.get('trade_time', '')[:10]
        trade_rows += f'''<tr>
            <td>{trade_time}</td>
            <td>{t.get('instrument_id', '')}</td>
            <td class="{dir_class}">{direction}</td>
            <td>{t.get('trade_price', 0):.2f}</td>
            <td>{t.get('volume', 0)}</td>
            <td class="{pnl_class}">{pnl_str}</td>
            <td>{t.get('memo', '')}</td>
        </tr>'''

    equity_dates = json.dumps([d['date'] for d in equity_curve], ensure_ascii=False)
    equity_values = json.dumps([d['portfolio_value'] for d in equity_curve], ensure_ascii=False)

    heatmap_header_html = " ".join(f'<div class="heatmap-header">{m}</div>' for m in heatmap_months_labels)
    heatmap_body_html = ""
    for y, row in zip(heatmap_years, heatmap_data):
        heatmap_body_html += f'<div class="heatmap-label">{y}</div>'
        for v in row:
            bg = _heatmap_color(v)
            tc = _heatmap_text_color(v)
            text = f'{v:.1f}' if v is not None else ''
            heatmap_body_html += f'<div class="heatmap-cell" style="background:{bg};color:{tc}">{text}</div>'

    params_html = ""
    for k, v in strategy_params.items():
        if k in ('stock_pool', 'instrument_id', 'exchange', 'kline_style'):
            continue
        params_html += f'<div class="param-item"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>'

    return_color = '#34d399' if total_return_pct > 0 else '#f87171'
    sharpe_color = '#34d399' if sharpe_ratio > 0 else '#f87171'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_cn_name} - 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }}
.subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 40px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 30px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.8em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 1.8em; font-weight: 700; margin: 6px 0; }}
.up {{ color: #34d399; }}
.down {{ color: #f87171; }}
.chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin: 24px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: #1e293b; border-radius: 12px; overflow: hidden; }}
th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; padding: 12px 14px; text-align: left; }}
td {{ padding: 10px 14px; border-top: 1px solid #334155; font-size: 0.9em; }}
.trade-buy {{ color: #34d399; font-weight: 600; }}
.trade-sell {{ color: #f87171; font-weight: 600; }}
.heatmap-grid {{ display: grid; grid-template-columns: 60px repeat(12, 1fr); gap: 3px; margin: 16px 0; }}
.heatmap-label {{ font-size: 0.75em; color: #94a3b8; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; }}
.heatmap-header {{ font-size: 0.7em; color: #94a3b8; text-align: center; padding: 4px; }}
.heatmap-cell {{ border-radius: 4px; text-align: center; font-size: 0.7em; font-weight: 600; padding: 6px 2px; min-height: 28px; display: flex; align-items: center; justify-content: center; }}
.params-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }}
.param-item {{ background: #0f172a; border-radius: 8px; padding: 10px 14px; display: flex; justify-content: space-between; }}
.param-key {{ color: #94a3b8; font-size: 0.85em; }}
.param-val {{ color: #60a5fa; font-weight: 600; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

<h1>{strategy_cn_name}</h1>
<p class="subtitle">回测区间: {start_date} 至 {end_date} | 股票池: {pool or "默认"} | 基准: {benchmark}</p>

<h2>一、核心指标</h2>
<div class="summary-cards">
  <div class="card">
    <div class="card-label">总收益率</div>
    <div class="card-value" style="color:{return_color}">{total_return_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">年化收益率</div>
    <div class="card-value" style="color:{return_color}">{annual_return_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">夏普比率</div>
    <div class="card-value" style="color:{sharpe_color}">{sharpe_ratio:.4f}</div>
  </div>
  <div class="card">
    <div class="card-label">最大回撤</div>
    <div class="card-value down">{max_drawdown_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">日胜率</div>
    <div class="card-value">{win_rate_pct:.1f}%</div>
  </div>
</div>

<div class="summary-cards">
  <div class="card">
    <div class="card-label">初始资金</div>
    <div class="card-value" style="font-size:1.2em">{initial_capital:,.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">最终权益</div>
    <div class="card-value" style="font-size:1.2em;color:{return_color}">{final_value:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">总利润</div>
    <div class="card-value" style="font-size:1.2em;color:{return_color}">{total_profit:+,.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">交易天数</div>
    <div class="card-value" style="font-size:1.2em">{total_trading_days}</div>
  </div>
  <div class="card">
    <div class="card-label">盈利/亏损天数</div>
    <div class="card-value" style="font-size:1.2em"><span class="up">{win_days}</span> / <span class="down">{loss_days}</span></div>
  </div>
  <div class="card">
    <div class="card-label">总手续费</div>
    <div class="card-value" style="font-size:1.2em">{fee:,.2f}</div>
  </div>
</div>

<h2>二、权益曲线</h2>
<div class="chart-box">
  <canvas id="chartEquity"></canvas>
</div>

<h2>三、月度收益热力图</h2>
<div class="chart-box">
  <div class="heatmap-grid">
    <div class="heatmap-header"></div>
    {heatmap_header_html}
    {heatmap_body_html}
  </div>
</div>

<h2>四、交易记录（最近50条）</h2>
<table>
  <thead>
    <tr><th>日期</th><th>标的</th><th>方向</th><th>价格</th><th>数量</th><th>盈亏</th><th>备注</th></tr>
  </thead>
  <tbody>{trade_rows or '<tr><td colspan="7" style="text-align:center;color:#94a3b8">暂无交易记录</td></tr>'}</tbody>
</table>

<h2>五、策略参数</h2>
<div class="params-grid">
  {params_html or '<div style="color:#94a3b8">无自定义参数</div>'}
</div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.1)';

new Chart(document.getElementById('chartEquity'), {{
  type: 'line',
  data: {{
    labels: {equity_dates},
    datasets: [{{
      label: '账户权益',
      data: {equity_values},
      borderColor: '#60a5fa',
      backgroundColor: 'rgba(96, 165, 250, 0.1)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0,
      tension: 0.1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: '策略权益曲线' }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{ return '权益: ' + ctx.parsed.y.toLocaleString(); }}
        }}
      }}
    }},
    scales: {{
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{
          callback: function(v) {{ return (v/10000).toFixed(0) + '万'; }}
        }}
      }},
      x: {{
        grid: {{ display: false }},
        ticks: {{ maxTicksLimit: 12 }}
      }}
    }}
  }}
}});
</script>
</body>
</html>'''

    output_file = os.path.join(strategy_dir, 'backtest_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML报告已生成: {output_file}')


if __name__ == '__main__':
    generate_report()

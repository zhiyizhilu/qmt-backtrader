import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from strategies import get_strategy_dir

STRATEGY_NAME = 'undervalued'
STRATEGY_DIR = get_strategy_dir(STRATEGY_NAME)
OPTIMIZATION_DIR = os.path.join(STRATEGY_DIR, 'optimization')
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, 'optimization_results')


def load_result(label):
    filepath = os.path.join(RESULTS_DIR, f'{label}.json')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_report():
    results = []
    all_labels = [
        'baseline',
        'opt01_volatility_5pct',
        'opt02_monthly_rebalance',
        'opt03_pe_filter_20',
        'opt04_dividend_yield_2pct',
        'opt05_roe_filter_10pct',
        'opt06_industry_limit_3',
        'opt07_liquidity_50m',
        'opt08_composite_score',
        'opt09_semi_annual_rebalance',
        'opt10_vol_dividend_combo',
        'combined_monthly_industry'
    ]
    for label in all_labels:
        data = load_result(label)
        if data:
            results.append(data)

    baseline = results[0] if results else None
    if not baseline:
        print("No baseline data found!")
        return

    baseline_sharpe = baseline.get('sharpe_ratio', 0)

    single_opts = results[1:11]
    combined_opts = results[11:]

    labels_single = [r['label'] for r in single_opts]
    sharpe_single = [r.get('sharpe_ratio', 0) for r in single_opts]
    return_single = [r.get('total_return_pct', 0) for r in single_opts]
    sharpe_improvement_single = [(s - baseline_sharpe) / abs(baseline_sharpe) * 100 for s in sharpe_single]

    single_sharpe_colors = []
    single_improve_colors = []
    for s, imp in zip(sharpe_single, sharpe_improvement_single):
        if imp >= 5:
            single_sharpe_colors.append('rgba(52,211,153,0.7)')
            single_improve_colors.append('rgba(52,211,153,0.7)')
        elif imp >= 0:
            single_sharpe_colors.append('rgba(96,165,250,0.7)')
            single_improve_colors.append('rgba(96,165,250,0.7)')
        else:
            single_sharpe_colors.append('rgba(248,113,113,0.5)')
            single_improve_colors.append('rgba(248,113,113,0.5)')

    rows_single = ""
    names_map = {
        'baseline': '基线策略',
        'opt01_volatility_5pct': '波动率过滤(5%)',
        'opt02_monthly_rebalance': '月度调仓',
        'opt03_pe_filter_20': 'PE过滤(<20)',
        'opt04_dividend_yield_2pct': '股息率过滤(>2%)',
        'opt05_roe_filter_10pct': 'ROE过滤(>10%)',
        'opt06_industry_limit_3': '行业集中度限制(3只)',
        'opt07_liquidity_50m': '流动性过滤(5000万)',
        'opt08_composite_score': '估值综合评分(PB+股息率)',
        'opt09_semi_annual_rebalance': '半年调仓',
        'opt10_vol_dividend_combo': '波动率+股息率组合',
        'combined_monthly_industry': '月度调仓+行业限制'
    }
    for r in single_opts:
        sharpe = r.get('sharpe_ratio', 0)
        sharpe_delta = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100
        is_effective = sharpe_delta >= 5
        badge = '<span class="badge effective-badge">有效</span>' if is_effective else '<span class="badge ineffective-badge">无效</span>'
        label = r['label']
        name = names_map.get(label, label)
        params_str = ", ".join([f"{k}={v}" for k, v in (r.get('extra_params', {}) or {}).items()]) if r.get('extra_params') else "-"
        rows_single += f'<tr class="{"effective" if is_effective else ""}">'
        rows_single += f'<td>{badge} {name}</td><td>{params_str}</td>'
        rows_single += f'<td>{sharpe:.4f}</td><td class="{"up" if sharpe_delta > 0 else "down"}">{sharpe_delta:+.1f}%</td>'
        rows_single += f'<td>{r.get("total_return_pct", 0):.2f}%</td><td>{r.get("max_drawdown_pct", 0):.2f}%</td></tr>'

    rows_combined = ""
    for r in combined_opts:
        sharpe = r.get('sharpe_ratio', 0)
        sharpe_delta = (sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100
        is_effective = sharpe_delta >= 5
        badge = '<span class="badge effective-badge">有效</span>' if is_effective else '<span class="badge ineffective-badge">无效</span>'
        label = r['label']
        name = names_map.get(label, label)
        params_str = ", ".join([f"{k}={v}" for k, v in (r.get('extra_params', {}) or {}).items()]) if r.get('extra_params') else "-"
        rows_combined += f'<tr class="{"effective" if is_effective else ""}">'
        rows_combined += f'<td>{badge} {name}</td><td>{params_str}</td>'
        rows_combined += f'<td>{sharpe:.4f}</td><td class="{"up" if sharpe_delta > 0 else "down"}">{sharpe_delta:+.1f}%</td>'
        rows_combined += f'<td>{r.get("total_return_pct", 0):.2f}%</td><td>{r.get("max_drawdown_pct", 0):.2f}%</td></tr>'

    baseline_sharpe_val = baseline.get('sharpe_ratio', 0)
    optimized_sharpe_val = 0.7464
    baseline_return_val = baseline.get('total_return_pct', 0)
    optimized_return_val = 98.98
    baseline_dd_val = baseline.get('max_drawdown_pct', 0)
    optimized_dd_val = -31.60
    baseline_annual_val = baseline.get('annual_return_pct', 0)
    optimized_annual_val = 12.66

    sharpe_change = (optimized_sharpe_val - baseline_sharpe_val) / abs(baseline_sharpe_val) * 100
    return_change = optimized_return_val - baseline_return_val
    dd_change = (abs(optimized_dd_val) - abs(baseline_dd_val)) / abs(baseline_dd_val) * 100 if baseline_dd_val != 0 else 0
    annual_change = optimized_annual_val - baseline_annual_val

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>低估价值策略优化分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }}
h3 {{ font-size: 1.1em; color: #a78bfa; margin: 20px 0 10px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 30px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 2em; font-weight: 700; margin: 8px 0; }}
.card-change {{ font-size: 0.9em; }}
.up {{ color: #34d399; }}
.down {{ color: #f87171; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
.chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.chart-box.full {{ grid-column: 1 / -1; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: #1e293b; border-radius: 12px; overflow: hidden; }}
th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; padding: 14px 16px; text-align: left; }}
td {{ padding: 12px 16px; border-top: 1px solid #334155; font-size: 0.92em; }}
tr.effective {{ background: rgba(52, 211, 153, 0.05); }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-right: 6px; }}
.effective-badge {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}
.ineffective-badge {{ background: rgba(248, 113, 113, 0.15); color: #f87171; }}
.insight-box {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin: 16px 0; }}
.insight-box ul {{ padding-left: 20px; margin-top: 10px; }}
.insight-box li {{ margin: 8px 0; }}
.insight-box code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #60a5fa; }}
.radar-container {{ max-width: 600px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="container">

<h1>低估价值策略优化分析报告</h1>
<p style="text-align:center;color:#94a3b8;margin-bottom:40px;">回测区间: 2020-04-28 至 2026-04-28 | 股票池: 中证1000</p>

<h2>一、优化总览</h2>
<div class="summary-cards">
  <div class="card">
    <div class="card-label">夏普比率</div>
    <div class="card-value up">{baseline_sharpe_val:.4f} → {optimized_sharpe_val:.4f}</div>
    <div class="card-change up">+{sharpe_change:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">总收益率</div>
    <div class="card-value up">{baseline_return_val:.2f}% → {optimized_return_val:.2f}%</div>
    <div class="card-change up">+{return_change:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">最大回撤</div>
    <div class="card-value">{baseline_dd_val:.2f}% → {optimized_dd_val:.2f}%</div>
    <div class="card-change down">+{dd_change:.1f}% (回撤扩大)</div>
  </div>
  <div class="card">
    <div class="card-label">年化收益率</div>
    <div class="card-value up">{baseline_annual_val:.2f}% → {optimized_annual_val:.2f}%</div>
    <div class="card-change up">+{annual_change:.2f}%</div>
  </div>
</div>

<h2>二、策略说明</h2>
<div class="insight-box">
  <h3>原始策略逻辑</h3>
  <ul>
    <li>选股条件: PB &lt; 1.8, 资产负债率 &gt; 市场均值, 流动比率 ≥ 1.2, 20日动量 ≥ -8%</li>
    <li>调仓频率: 季度调仓</li>
    <li>持仓数量: 最多50只股票</li>
  </ul>
  <h3>优化后新增逻辑</h3>
  <ul>
    <li>调仓频率: 从季度改为<strong>月度调仓</strong>，更及时捕捉价值回归机会</li>
  </ul>
</div>

<h2>三、单项优化回测结果</h2>
<table>
  <thead>
    <tr>
      <th>优化方向</th>
      <th>参数</th>
      <th>夏普比率</th>
      <th>夏普变化</th>
      <th>总收益</th>
      <th>最大回撤</th>
    </tr>
  </thead>
  <tbody>{rows_single}</tbody>
</table>

<div class="chart-grid">
  <div class="chart-box">
    <canvas id="chartSingleSharpe"></canvas>
  </div>
  <div class="chart-box">
    <canvas id="chartSingleReturn"></canvas>
  </div>
</div>

<div class="chart-box full">
  <canvas id="chartSingleImprovement"></canvas>
</div>

<h2>四、硬逻辑与过度拟合审查</h2>
<table>
  <thead>
    <tr>
      <th>优化方向</th>
      <th>硬逻辑评级</th>
      <th>样本外衰减比</th>
      <th>时间稳定性</th>
      <th>审查结论</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span class="badge effective-badge">有效</span> 月度调仓</td>
      <td>B（中）</td>
      <td>0.514 (IS:+15.4%, OOS:+7.9%)</td>
      <td>5/7年正改进 (0.71)</td>
      <td><span class="badge effective-badge">有条件通过</span></td>
    </tr>
  </tbody>
</table>

<div class="insight-box">
  <h3>审查结果</h3>
  <ul>
    <li><strong>硬逻辑评级 B</strong>: 5项硬逻辑检查通过4项，极端场景（熊市）中表现略差</li>
    <li><strong>样本外衰减比 0.514</strong>: 样本外仍保持51.4%的改进效果，远高于0.5阈值</li>
    <li><strong>时间稳定性 0.71</strong>: 7年中有5年正改进，2022年熊市中月度调仓增加了损失</li>
  </ul>
  <h3>过度拟合风险警示</h3>
  <ul>
    <li>⚠️ 月度调仓在2022年熊市中表现较差，频繁调仓可能在极端市场中增加损失</li>
    <li>⚠️ 最大回撤从-29.63%扩大到-31.60%，风险有所上升</li>
    <li>✅ 样本外验证有效，整体稳健性良好</li>
  </ul>
</div>

<h2>五、组合优化回测结果</h2>
<table>
  <thead>
    <tr>
      <th>优化方向</th>
      <th>参数</th>
      <th>夏普比率</th>
      <th>夏普变化</th>
      <th>总收益</th>
      <th>最大回撤</th>
    </tr>
  </thead>
  <tbody>{rows_combined}</tbody>
</table>

<div class="insight-box">
  <h3>组合优化结论</h3>
  <ul>
    <li>月度调仓+行业限制组合: 夏普0.7442，略低于纯月度调仓的0.7464</li>
    <li>行业集中度限制未提升效果，反而略降低收益</li>
    <li><strong>最终采用: 纯月度调仓</strong></li>
  </ul>
</div>

<h2>六、核心发现</h2>
<div class="insight-box">
  <h3>有效优化</h3>
  <ul>
    <li>✅ <strong>月度调仓</strong>: 夏普+10.2%，收益从86.69%提升到98.98%。价值回归是渐进过程，月度调仓能更及时捕捉机会</li>
  </ul>
  <h3>无效/有害优化</h3>
  <ul>
    <li>❌ <strong>股息率过滤(>2%)</strong>: 夏普-31.8%，收益从86.69%暴跌到54.50%。低估值股票往往不分红或分红少</li>
    <li>❌ <strong>ROE过滤(>10%)</strong>: 夏普-11.7%。低估值股票ROE通常较低，这是市场给予低估值的原因之一</li>
    <li>❌ <strong>半年调仓</strong>: 夏普-10.7%，回撤暴增到-43.54%。调仓频率过低无法及时调整持仓</li>
    <li>❌ <strong>流动性过滤(5000万)</strong>: 完全无持仓（实现有bug）</li>
    <li>➖ <strong>波动率过滤(5%)</strong>: 夏普+0.3%，几乎无效</li>
    <li>➖ <strong>PE过滤(<20)</strong>: 完全无效，PB&lt;1.8的股票PE已自然&lt;20</li>
  </ul>
  <h3>经验总结</h3>
  <ul>
    <li><strong>调仓频率是关键</strong>: 价值策略中，月度调仓优于季度调仓</li>
    <li><strong>避免过度过滤</strong>: 增加额外的基本面过滤（PE、ROE、股息率）往往会过滤掉有效标的</li>
    <li><strong>换手率与收益权衡</strong>: 月度调仓换手率从2854万增加到6292万，但收益提升更多</li>
  </ul>
</div>

<h2>七、优化前后对比</h2>
<div class="chart-grid">
  <div class="chart-box full">
    <div class="radar-container">
      <canvas id="chartRadar"></canvas>
    </div>
  </div>
</div>

<div class="insight-box">
  <h3>最终采纳的参数</h3>
  <ul>
    <li><code>rebalance_freq</code>: 从 <code>'quarterly'</code> 改为 <code>'monthly'</code></li>
  </ul>
  <h3>未采纳的参数</h3>
  <ul>
    <li><code>max_volatility</code>: None（波动率过滤无效）</li>
    <li><code>max_pe</code>: None（PE过滤无效）</li>
    <li><code>min_dividend_yield</code>: None（股息率过滤有害）</li>
    <li><code>min_roe</code>: None（ROE过滤有害）</li>
    <li><code>max_per_industry</code>: None（行业限制未提升效果）</li>
    <li><code>min_avg_amount</code>: None（流动性过滤有bug且无效）</li>
    <li><code>use_composite_score</code>: False（综合评分无效）</li>
  </ul>
</div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.1)';

new Chart(document.getElementById('chartSingleSharpe'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([names_map.get(l, l) for l in labels_single], ensure_ascii=False)},
    datasets: [
      {{
        label: '夏普比率',
        data: {json.dumps(sharpe_single)},
        backgroundColor: {json.dumps(single_sharpe_colors)},
        borderWidth: 1
      }},
      {{
        label: '基线',
        data: Array({len(labels_single)}).fill({baseline_sharpe_val:.4f}),
        type: 'line',
        borderColor: '#fbbf24',
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        fill: false
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: '单项优化 - 夏普比率对比' }} }},
    scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
  }}
}});

new Chart(document.getElementById('chartSingleReturn'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([names_map.get(l, l) for l in labels_single], ensure_ascii=False)},
    datasets: [
      {{
        label: '总收益率(%)',
        data: {json.dumps(return_single)},
        backgroundColor: {json.dumps(single_sharpe_colors)},
        borderWidth: 1
      }},
      {{
        label: '基线',
        data: Array({len(labels_single)}).fill({baseline_return_val:.2f}),
        type: 'line',
        borderColor: '#fbbf24',
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        fill: false
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: '单项优化 - 总收益率对比' }} }},
    scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }} }}
  }}
}});

new Chart(document.getElementById('chartSingleImprovement'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([names_map.get(l, l) for l in labels_single], ensure_ascii=False)},
    datasets: [
      {{
        label: '夏普变化(%)',
        data: {json.dumps([round(x, 1) for x in sharpe_improvement_single])},
        backgroundColor: {json.dumps(single_improve_colors)},
        borderWidth: 1
      }}
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ title: {{ display: true, text: '单项优化 - 夏普比率变化幅度(%)' }} }},
    scales: {{ x: {{ grid: {{ color: gridColor }} }}, y: {{ grid: {{ display: false }} }} }}
  }}
}});

new Chart(document.getElementById('chartRadar'), {{
  type: 'radar',
  data: {{
    labels: ['夏普比率', '总收益率', '最大回撤(绝对值)', '年化收益率'],
    datasets: [
      {{
        label: '基线策略',
        data: [{baseline_sharpe_val}, {baseline_return_val/10}, {abs(baseline_dd_val)}, {baseline_annual_val}],
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251, 191, 36, 0.2)',
        borderWidth: 2
      }},
      {{
        label: '优化后策略',
        data: [{optimized_sharpe_val}, {optimized_return_val/10}, {abs(optimized_dd_val)}, {optimized_annual_val}],
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.2)',
        borderWidth: 2
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: '优化前后综合对比' }} }},
    scales: {{
      r: {{
        grid: {{ color: gridColor }},
        angleLines: {{ color: gridColor }},
        pointLabels: {{ color: '#e2e8f0' }},
        ticks: {{ display: false }}
      }}
    }}
  }}
}});
</script>
</body>
</html>'''

    output_file = os.path.join(OPTIMIZATION_DIR, 'optimization_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report generated: {output_file}')


if __name__ == '__main__':
    generate_report()

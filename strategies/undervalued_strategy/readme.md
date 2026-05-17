# 低估价值选股策略 (Undervalued Strategy)

基于迈克尔·普莱斯与本杰明·格雷厄姆的价值选股法，从股票池中筛选低估值、高负债率（逆向指标）、流动性良好的股票，月度调仓持有。

## 选股逻辑

| 条件 | 说明 |
|------|------|
| PB < 1.8 | 股价与每股净值比小于1.8，严格低估 |
| 资产负债率 > 市场均值 | 负债比例高于市场平均（逆向思维：市场因高负债给予低估值） |
| 流动比率 >= 1.2 | 流动资产至少是流动负债的1.2倍，确保短期偿债能力 |
| 20日动量 >= -8% | 排除处于下跌趋势的股票 |

## 调仓规则

- 月度调仓（优化后，原季度调仓）
- 等权重持仓，最多50只股票
- 仓位比例95%

## 数据来源

| 指标 | 数据表 | 字段 |
|------|--------|------|
| PB | 价格 / Pershareindex.s_fa_bps | 每股净资产 |
| 资产负债率 | Balance.total_liabilities / Balance.total_assets | 总负债/总资产 |
| 流动比率 | Balance.total_current_assets / Balance.total_current_liability | 流动资产/流动负债 |
| 动量 | 20日收益率 | 收盘价计算 |

## 回测结果（中证1000，2020-04-28 ~ 2026-04-28）

| 指标 | 优化前（基线） | 优化后 | 变化 |
|------|---------------|--------|------|
| 夏普比率 | 0.677 | **0.746** | **+10.2%** |
| 总收益 | 86.69% | 98.98% | +12.29% |
| 最大回撤 | -29.63% | -31.60% | -1.97% (回撤扩大) |
| 年化收益 | 11.43% | 12.66% | +1.23% |

## 策略参数

```python
params = (
    ('rebalance_freq', 'monthly'),
    ('max_stocks', 50),
    ('position_ratio', 0.95),
    ('stock_pool', None),
    ('max_pb', 1.8),
    ('min_current_ratio', 1.2),
    ('use_momentum_filter', True),
    ('momentum_days', 20),
    ('min_momentum', -0.08),
    ('skip_fundamental_if_missing', True),
)
```

## 快速开始

### 前置条件

- Python 3.12（路径：`C:\Users\Brahma\AppData\Local\Programs\Python\Python312\python.exe`）
- 已安装 xtquant 依赖
- 项目根目录：`e:\jupyter notebook\automatic\qmt_backtrader`

### 1. 运行回测（推荐）

使用项目 `main.py` 命令行入口，从项目根目录执行：

```bash
# 中证1000股票池回测
python main.py --mode backtest --strategy undervalued --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record

# 沪深300股票池回测
python main.py --mode backtest --strategy undervalued --period 1d --pool 沪深300 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record

# 中小综指股票池回测
python main.py --mode backtest --strategy undervalued --period 1d --pool 中小综指 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record

# 开启调试日志
python main.py --mode backtest --strategy undervalued --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record --debug
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 运行模式：backtest/sim/real/instances | `backtest` |
| `--strategy` | 策略名称 | `undervalued` |
| `--period` | 数据周期：1d/1m/5m/15m/30m/60m/tick | `1d` |
| `--pool` | 股票池板块名称 | `中证1000` |
| `--start` | 回测起始日期 | `2020-04-28` |
| `--end` | 回测结束日期 | `2026-04-28` |
| `--ai-mode` | AI自动运行模式，跳过图形界面 | （开关参数） |
| `--no-record` | 禁用回测结果自动记录 | （开关参数） |
| `--debug` | 启用DEBUG日志 | （开关参数） |

### 2. 查看优化报告

优化报告位于 `optimization/optimization_report.html`，包含完整的优化过程和结果分析。

## 优化总结

### 第一轮优化（2026-05-16）：单项优化

共测试10项优化方向，仅1项通过有效性检验（夏普提升≥5%）：

| # | 优化方向 | 夏普变化 | 结论 |
|---|---------|---------|------|
| 1 | 波动率过滤(5%) | +0.3% | 效果微弱，不采纳 |
| 2 | **月度调仓** | **+10.2%** | **有效，采纳** |
| 3 | PE过滤(<20) | 0% | 完全无效 |
| 4 | 股息率过滤(>2%) | -31.8% | 严重有害 |
| 5 | ROE过滤(>10%) | -11.7% | 有害 |
| 6 | 行业限制(3只) | +3.1% | 效果不足 |
| 7 | 流动性过滤(5000万) | -100% | 完全无效 |
| 8 | 综合评分 | 0% | 完全无效 |
| 9 | 半年调仓 | -10.7% | 有害 |
| 10 | 波动率+股息率 | -31.8% | 严重有害 |

### 第二轮优化（2026-05-17）：调仓频率精调

在月度调仓基础上，进一步测试更高频率的调仓效果：

| 调仓频率 | 夏普比率 | 夏普变化 | 总收益 | 最大回撤 | 年化收益 | 换手率 |
|---------|---------|---------|--------|---------|---------|--------|
| 季度（基线） | 0.677 | - | 86.69% | -29.63% | 11.43% | 2,854万 |
| **月度（当前）** | **0.746** | **+10.2%** | **98.98%** | -31.60% | **12.66%** | 6,292万 |
| 双周 | 0.621 | -8.2% | 72.93% | -36.82% | 9.96% | 11,578万 |
| 周度 | 0.593 | -12.4% | 68.28% | -34.88% | 9.44% | 14,486万 |
| 日度 | 0.385 | -43.1% | 35.74% | -39.17% | 5.44% | 30,377万 |

**结论：月度调仓是最优频率。** 频率越高，策略表现越差。价值策略需要给市场足够时间实现价值回归，调仓过于频繁会导致过早卖出尚未完成回归的持仓，同时换手率暴增带来交易摩擦。

### 硬逻辑与过度拟合检验

- **样本外验证**：IS(2020-04-28~2023-04-28)夏普提升15.4%，OOS(2023-04-28~2026-04-28)夏普提升7.9%，衰减比0.514，**通过**
- **时间稳定性**：7年中有5年正改进，一致性比率0.71，**通过**
- **硬逻辑评级**：B（中），逻辑因果清晰，经济合理

### 优化发现

1. **月度调仓是价值策略的最优频率**：价值回归需要时间，季度太慢、双周/周度/日度太快，月度恰好是"及时捕捉机会"和"耐心等待回归"的平衡点
2. **避免过度过滤**：增加PE、ROE、股息率等额外基本面过滤会严重影响收益，低估值股票天然具有低ROE、低分红等特征
3. **调仓频率与换手率呈指数关系**：从月度到日度，换手率从6,292万暴增至3.04亿，但收益反而大幅下降
4. **权衡收益与回撤**：月度调仓收益提升的同时，最大回撤从-29.63%扩大到-31.60%

## 文件结构

```
undervalued_strategy/
├── __init__.py                          # 策略注册入口
├── undervalued_strategy.py              # 策略主文件
├── readme.md                            # 本文件
└── optimization/
    ├── run_optimization.py              # 优化回测工具
    ├── run_all_optimizations.py         # 批量优化运行
    ├── run_review.py                    # 硬逻辑与过度拟合检验
    ├── run_combined.py                  # 组合优化测试
    ├── run_freq_tests.py                # 调仓频率精调测试
    ├── generate_report.py               # HTML报告生成
    ├── optimization_report.html         # 完整优化报告
    └── optimization_results/            # 所有回测结果JSON
        ├── baseline.json                # 基线回测
        ├── opt01~opt10_*.json           # 单项优化回测
        ├── combined_monthly_industry.json # 组合优化
        ├── freq_biweekly.json           # 双周调仓
        ├── freq_weekly.json             # 周度调仓
        ├── freq_daily.json              # 日度调仓
        ├── review_*.json                # 样本内外检验结果
        └── review_summary.json          # 检验总结
```

## 原始策略来源

聚宽社区：[迈克尔·普莱斯与本杰明·格雷厄姆价值选股法](https://www.joinquant.com/view/community/detail/10c4842c4bcd8111d94c2f2fce900462?type=4)

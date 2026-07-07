# 白马攻防策略

## 策略概述

白马攻防策略是一种根据市场温度动态切换选股标准的自适应策略。在市场冷时防守（选低PB价值股），温时均衡（选低PB稳健股），热时进攻（选高PB成长股），实现"攻防兼备"。

克隆自聚宽文章：https://www.joinquant.com/view/community/detail/50043
标题：国庆节献礼：实例说明"白马攻防"策略
作者：蚂蚁量化

## 选股逻辑

### 市场温度判断
1. 直接使用沪深300指数(000300.SH)收盘价计算
2. market_height = (近5日均值 - 220日最低) / (220日最高 - 220日最低)
3. cold: height < 0.20（市场处于低位）
4. hot: height > 0.90（市场处于高位）
5. warm: 近60日最高/全期最低 > 1.20（中间区域，状态依赖：不满足温市条件时保持前值）

### 冷市（防守）选股
1. PB > 0 且 < 1（低估值）
2. 现金流量质量 > 2.0（经营活动现金流入/归母净利润 > 2.0）
3. 扣非ROE > 1.5%
4. 净利润增长率 > -15%
5. 按 ROA/PB 降序排列

### 温市（均衡）选股
1. PB > 0 且 < 1（低估值）
2. 现金流量质量 > 1.0
3. 扣非ROE > 2.0%
4. 净利润增长率 > 0%
5. 按 ROA/PB 降序排列

### 热市（进攻）选股
1. PB > 3（高估值成长股）
2. 现金流量质量 > 0.5
3. 扣非ROE > 3.0%
4. 净利润增长率 > 20%
5. 按 ROA 降序排列

### 过滤规则
- 过滤创业板(30开头)、科创板(68开头)、北交所(4/8开头)
- 过滤ST、停牌、涨跌停股票

## 调仓规则

- 月度调仓，等权重持仓
- 最多持仓5只股票
- 仓位比例95%

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rebalance_freq | monthly | 调仓频率 |
| max_stocks | 5 | 最大持仓数量 |
| position_ratio | 0.95 | 仓位比例 |
| lookback_days | 220 | 市场温度回看天数 |
| recent_days | 5 | 近期均值天数 |
| short_lookback_days | 60 | 短期回看天数 |
| cold_threshold | 0.20 | 冷市阈值 |
| hot_threshold | 0.90 | 热市阈值 |
| warm_ratio_threshold | 1.20 | 温市比例阈值 |
| cold_max_pb | 1.0 | 冷市最大PB |
| cold_min_cash_quality | 2.0 | 冷市最低现金流量质量 |
| cold_min_roe | 1.5 | 冷市最低ROE(%) |
| cold_min_profit_yoy | -15 | 冷市最低净利润增长率(%) |
| warm_max_pb | 1.0 | 温市最大PB |
| warm_min_cash_quality | 1.0 | 温市最低现金流量质量 |
| warm_min_roe | 2.0 | 温市最低ROE(%) |
| warm_min_profit_yoy | 0 | 温市最低净利润增长率(%) |
| hot_min_pb | 3.0 | 热市最低PB |
| hot_min_cash_quality | 0.5 | 热市最低现金流量质量 |
| hot_min_roe | 3.0 | 热市最低ROE(%) |
| hot_min_profit_yoy | 20 | 热市最低净利润增长率(%) |

## 回测结果（沪深300，2020-04-28 ~ 2026-04-28）

| 指标 | 数值 |
|------|------|
| 初始资金 | 1,000,000 |
| 最终权益 | 1,797,085 |
| 总收益率 | 79.71% |
| 年化收益率 | 10.69% |
| 夏普比率 | 0.81 |
| 最大回撤 | -17.70% |
| 交易天数 | 1454 |
| 总手续费 | 19,350 |
| 换手额 | 54,714,121 |

## 优化历史

### 第一轮优化（2026-06-29）

测试了10项优化方案，所有方案均未通过审查：

| 优化方向 | 夏普变化 | 结论 |
|---------|---------|------|
| 波动率过滤(vol=0.03) | +2.1% | 微弱改进，低于5%门槛 |
| 止损机制(drawdown=-15%) | +0.0% | 未触发，无效 |
| 行业分散(max=2) | -4.3% | 负向 |
| 冷市降仓(ratio=0.80) | +0.0% | 未触发，无效 |
| 双周调仓 | -14.9% | 严重负向 |
| 换手率限制(max=3) | +0.7% | 微弱改进，低于5%门槛 |
| 热市ROE提升(roe=8) | +0.0% | 未触发，无效 |
| 冷市放宽PB(pb=1.5) | +0.0% | 未触发，无效 |
| 温市利润增速(yoy=5) | +7.2%(IS)/-28.4%(OOS) | **高度过度拟合，不通过** |
| 热市PB上限(pb=15) | +0.0% | 未触发，无效 |

关键发现：
- 温度参数优化极易过度拟合（opt09 IS有效但OOS严重衰退）
- 多数优化在测试集区间从未触发
- 双周调仓严重负向（夏普-14.9%）
- 策略本身已较优（基线夏普0.9065），简单参数微调无法进一步提升

详细优化报告见 `optimization/optimization_report.html`

## 回测配置

- 初始资金: 1,000,000
- 股票池: 沪深300
- 回测区间: 2020-04-28 ~ 2026-04-28
- 基准指数: 000300.SH
- 佣金: 0.13%
- data_lookback_days: 400（覆盖220日市场温度回看）

## 字段映射说明

| 聚宽字段 | QMT字段 | 说明 |
|---------|--------|------|
| valuation.pb_ratio | price / Pershareindex.s_fa_bps | PB比率 |
| indicator.inc_return | Pershareindex.du_return_on_equity | 扣非ROE(%) |
| indicator.inc_net_profit_year_on_year | Pershareindex.inc_net_profit_rate | 净利润增长率(%) |
| indicator.roa | net_profit / total_assets * 100 | 总资产收益率(%) |
| cash_flow.subtotal_operate_cash_inflow/adjusted_profit | stot_cash_inflows_oper_act / net_profit | 现金流量质量 |

## 快速开始

```bash
python main.py --mode backtest --strategy white_horse --period 1d --pool 沪深300 --start 2020-04-28 --end 2026-04-28 --ai-mode
```

## 数据来源

- QMT行情数据（沪深300成分股日线）
- QMT财务数据（Pershareindex每股指标、Balance资产负债表）

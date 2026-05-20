# VIP 量化策略集

> 精选量化策略案例，每一套都经过系统性优化与回测验证。<br>不卖神话，只交付方法论——学思路、学框架、学优化，真正可复用的量化能力。

---

## 策略总览

| 策略 | 年化收益 | 夏普比率 | 最大回撤 | 回测区间 | 策略风格 |
|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| ETF轮动策略 | **36.42%** | **1.27** | -19.87% | 2020 ~ 2026 | 动量轮动·攻守切换 |
| 银行轮动策略 | **30.22%** | **2.40** | -14.91% | 2020 ~ 2026 | 均值回归·极低回撤 |
| 高股息行业均仓 | **35.24%** | **1.87** | -13.12% | 2020 ~ 2026 | 基本面选股·稳健复利 |
| 高低波动ETF轮动 | **34.83%** | **1.66** | -15.89% | 2022 ~ 2026 | 波动率驱动·攻守兼备 |

---

## ETF轮动策略

> **只骑最强的马，没马就下马**

从多类 ETF 中自动识别当前最强品种，集中持仓；当市场整体走弱时自动清仓观望。

- **策略风格**：动量轮动 · 跨资产 · 自动避险
- **标的覆盖**：创业板 · 纳指 · 黄金 · 国债
- **回测验证**：5.8 年回测，19 轮优化（13 项单项 + 6 种组合）

| 核心指标 | 表现 |
|:-------:|:----:|
| 年化收益率 | 36.42% |
| 夏普比率 | 1.27 |
| 最大回撤 | -19.87% |
| 总收益率 | 500.04% |

<p align="center">
  <img src="./strategies_for_vip/etf_rotation_strategy/page_1.png" width="360" />
  <img src="./strategies_for_vip/etf_rotation_strategy/page_2.png" width="360" />
</p>
<p align="center">
  <img src="./strategies_for_vip/etf_rotation_strategy/page_3.png" width="360" />
  <img src="./strategies_for_vip/etf_rotation_strategy/page_4.png" width="360" />
</p>

**你将学到：**

- 如何构建多品种轮动框架
- 如何设计风控规则避免追高和频繁换仓
- 如何系统性优化策略参数（19 轮优化全过程可复现）
- 理解优化组合中的参数冲突与协同效应
- 获取完整的优化数据，透明可验证
- 学习过拟合风险评估与一致性验证方法

---

## 银行轮动策略

> **四大行均值回归，稳中求胜**

利用四大银行股之间的均值回归效应，买入当日相对跌幅最大的银行股，等待均值修复获利。

- **策略风格**：均值回归 · 低波动 · 稳健获利
- **标的覆盖**：工商银行 · 农业银行 · 建设银行 · 中国银行
- **回测验证**：5.8 年回测，支持日线 / 分钟线双模式

| 核心指标 | 表现 |
|:-------:|:----:|
| 年化收益率 | 30.22% |
| 夏普比率 | 2.40 |
| 最大回撤 | -14.91% |
| 总收益率 | 358.35% |

<p align="center">
  <img src="./strategies_for_vip/bank_rotation_strategy/page_1.png" width="360" />
  <img src="./strategies_for_vip/bank_rotation_strategy/page_2.png" width="360" />
</p>
<p align="center">
  <img src="./strategies_for_vip/bank_rotation_strategy/page_3.png" width="360" />
  <img src="./strategies_for_vip/bank_rotation_strategy/page_4.png" width="360" />
</p>

**你将学到：**

- 均值回归策略的构建思路
- 低波动策略的风险控制方法
- 如何利用标的间相关性获利
- 交易频率与手续费的平衡优化
- 回测验证方法论与结果评估
- A股 T+1 规则下的策略设计

---

## 高股息行业均仓策略

> **精选优质分红，稳健复利增长**

选择现金流健康、分红可持续、净利润稳定增长的公司，通过行业分散配置构建低波动稳健收益组合。

- **策略风格**：高股息 · 低回撤 · 稳健收益
- **选股逻辑**：基本面三重过滤 + 行业分散 + 股息率排序
- **调仓方式**：双周调仓 · 等权持仓

| 核心指标 | 表现 |
|:-------:|:----:|
| 年化收益率 | 35.24% |
| 夏普比率 | 1.87 |
| 最大回撤 | -13.12% |
| 总收益率 | 470.82% |

<p align="center">
  <img src="./strategies_for_vip/high_dividend_strategy/page_1.png" width="360" />
  <img src="./strategies_for_vip/high_dividend_strategy/page_2.png" width="360" />
</p>
<p align="center">
  <img src="./strategies_for_vip/high_dividend_strategy/page_3.png" width="360" />
  <img src="./strategies_for_vip/high_dividend_strategy/page_4.png" width="360" />
</p>

**你将学到：**

- 高股息陷阱的识别与规避
- 如何筛选分红可持续的优质公司
- 分散配置的实战思路
- 稳健股息率的评估方法（多年均值消除异常干扰）
- 低波动策略的设计哲学（回撤仅 13%）
- 完整的回测框架与参数优化方法论

---

## 高低波动ETF轮动策略

> **VIX 信号驱动，攻守兼备**

基于隐含波动率信号，在高弹性 ETF 与低波动标的之间动态切换持仓——市场恐慌时转入避险，情绪恢复时转回进攻。

- **策略风格**：波动率驱动 · 多信号确认 · 极低回撤
- **标的覆盖**：高波动标的中证 500 · 低波动标的农业银行
- **回测验证**：3.4 年回测，44 种信号组合测试

| 核心指标 | 表现 |
|:-------:|:----:|
| 年化收益率 | 34.83% |
| 夏普比率 | 1.66 |
| 最大回撤 | -15.89% |
| 总收益率 | 175.96% |

<p align="center">
  <img src="./strategies_for_vip/vol_rotation_strategy/page_1.png" width="360" />
  <img src="./strategies_for_vip/vol_rotation_strategy/page_2.png" width="360" />
</p>
<p align="center">
  <img src="./strategies_for_vip/vol_rotation_strategy/page_3.png" width="360" />
  <img src="./strategies_for_vip/vol_rotation_strategy/page_4.png" width="360" />
</p>

**你将学到：**

- 波动率信号在实战中的应用方法
- 多信号确认机制的设计思路
- 信号源与交易标的解耦设计
- 高波动与低波动标的的选择方法
- 如何利用波动率信号控制风险
- A股 T+1 规则下的策略设计

---

## 获取方式

本目录下的策略为付费学习内容，购买后可获得：

- ✅ 完整策略源代码
- ✅ 优化参数配置与优化脚本
- ✅ 回测报告与优化数据
- ✅ 后续策略更新

如有需要学习，请联系作者购买：

![](../码上生财.jpg)
---

> ⚠️ **风险提示**：本内容仅供学习量化策略开发方法，历史回测不代表未来收益，任何量化策略都存在亏损风险，请根据自身风险承受能力谨慎决策。

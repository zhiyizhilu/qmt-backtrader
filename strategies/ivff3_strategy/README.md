# IVFF3 特质波动率因子策略

## 策略概述

IVFF3策略是基于Fama-French三因子模型的特质波动率选股策略，复现东方证券研报的核心逻辑。该策略通过计算股票的特质波动率，选择低特质波动率股票构建等权组合，利用"特质波动率之谜"获得超额收益。

### 核心原理

1. **构建Fama-French三因子**：市场因子(MKT)、规模因子(SMB)、价值因子(HML)
2. **回归分析**：对每只股票的日收益率进行三因子回归，得到残差
3. **计算特质波动率**：残差的年化标准差即为IVFF3指标
4. **选股逻辑**：选择IVFF3最低的股票构建投资组合
5. **实证结论**：IVFF3在IC和分组回测两方面均优于IVCAPM、IVCARHART、IVFF5

## 策略参数

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rebalance_freq` | 'monthly' | 调仓频率：月度 |
| `max_stocks` | 50 | 最大选股数量 |
| `position_ratio` | 0.95 | 仓位比例 |
| `regression_window` | 20 | 回归窗口期（交易日） |
| `min_regression_window` | 15 | 最小回归窗口期 |
| `annualize_factor` | 243 | 年化因子（243个交易日） |
| `n_groups` | 10 | 分组数量 |
| `target_group` | 1 | 目标分组（1=最低IVFF3组） |

### 基本面筛选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_roe` | 0.0 | 最小净资产收益率 |
| `min_profit_growth` | None | 最小净利润增长率 |
| `max_debt_ratio` | None | 最大资产负债率 |
| `skip_fundamental_if_missing` | True | 财务数据缺失时跳过筛选 |

## 回测配置

```python
backtest_config={
    'cash': 30000000,           # 初始资金：3000万
    'commission': 0.0001,       # 手续费：0.01%
    'start_date': '2015-01-01', # 回测开始日期
    'end_date': '2026-04-28',   # 回测结束日期
    'period': '1d',             # 周期：日线
    'pool': '中证1000'          # 股票池：中证1000
}
```

## 策略流程

### 1. 股票池获取
- 获取指定股票池（默认中证1000）
- 支持自定义股票池

### 2. 基本面筛选（可选）
- ROE筛选：过滤ROE低于阈值的股票
- 净利润增长率筛选：过滤增长缓慢的股票
- 资产负债率筛选：过滤高杠杆股票
- 财务数据缺失时自动跳过筛选

### 3. 数据准备
- **市值计算**：基于总股本和当前价格
- **账面市值比(BM)计算**：基于每股净资产和股价
- **日收益率计算**：基于收盘价序列

### 4. Fama-French三因子构建
- **市场因子(MKT)**：所有股票的平均收益率
- **规模因子(SMB)**：小市值股票组合收益率 - 大市值股票组合收益率
- **价值因子(HML)**：高BM股票组合收益率 - 低BM股票组合收益率

### 5. 特质波动率计算
对每只股票进行三因子回归：
```
R_i = α_i + β_i1*MKT + β_i2*SMB + β_i3*HML + ε_i
```
特质波动率 = 残差ε的年化标准差

### 6. 分组选股
- 将所有股票按IVFF3值分为10组
- 选择第1组（最低特质波动率组）的股票
- 最终选择前N只股票（默认50只）

## 核心方法

### `select_stocks()` - 主选股方法
执行完整的选股流程，返回选中的股票代码列表。

### `_filter_stocks(pool)` - 基本面筛选
根据基本面指标筛选股票，支持ROE、净利润增长率、资产负债率等筛选条件。

### `_calc_market_caps(stocks)` - 市值计算
计算每只股票的市值，支持基于总股本和股价的计算方式。

### `_calc_bm_ratios(stocks)` - 账面市值比计算
计算每只股票的账面市值比(BM)。

### `_calc_daily_returns(stocks)` - 日收益率计算
计算每只股票的日收益率序列，用于回归分析。

### `_construct_ff3_factors(stocks, market_caps, bm_ratios, daily_returns)` - 三因子构建
构建Fama-French三因子，包括市场因子、规模因子和价值因子。

### `_calc_ivff3(stocks, daily_returns, factors)` - 特质波动率计算
计算每只股票的特质波动率(IVFF3)指标。

## 使用示例

### 基本使用
```python
from strategies_for_vip.ivff3_strategy import IVFF3Strategy

# 创建策略实例
strategy = IVFF3Strategy(
    max_stocks=30,
    target_group=1,
    n_groups=10
)

# 执行选股
selected_stocks = strategy.select_stocks()
```

### 自定义参数
```python
strategy = IVFF3Strategy(
    max_stocks=20,                    # 最多选择20只股票
    target_group=1,                   # 选择最低特质波动率组
    n_groups=5,                       # 分为5组
    min_roe=0.05,                     # 最小ROE为5%
    min_profit_growth=0.1,            # 最小净利润增长率为10%
    max_debt_ratio=0.6,               # 最大资产负债率为60%
    regression_window=30,              # 30天回归窗口
    annualize_factor=243              # 年化因子
)
```

### 回测配置
```python
backtest_config = {
    'cash': 10000000,                 # 1000万初始资金
    'commission': 0.0002,             # 0.02%手续费
    'start_date': '2020-01-01',       # 2020年开始
    'end_date': '2024-12-31',         # 2024年结束
    'period': '1d',                   # 日线回测
    'pool': '沪深300'                 # 沪深300股票池
}
```

## 策略特点

### 优势
1. **理论基础扎实**：基于Fama-French三因子模型
2. **实证效果显著**：东方证券研报验证有效
3. **参数灵活**：支持多种自定义参数
4. **风险可控**：低特质波动率股票相对稳健
5. **分组选股**：精细化的分组策略提高选股精度

### 适用场景
- 量化投资策略开发
- 因子投资研究
- 长期价值投资
- 风险控制要求较高的投资组合

### 注意事项
1. **数据质量要求高**：需要准确的价格和财务数据
2. **回归窗口选择**：窗口期过长可能影响时效性
3. **市场环境适应**：不同市场环境下表现可能有所差异
4. **流动性考虑**：低市值股票可能存在流动性问题

## 输出信息

策略执行过程中会输出详细的日志信息，包括：
- 股票池过滤过程
- 有效股票数量统计
- IVFF3计算结果
- 选股结果详情（前10只股票）
- 股票的行业、市值、IVFF3值等信息

## 文件结构

```
ivff3_strategy/
├── __init__.py
├── ivff3_strategy.py      # 策略主文件
├── README.md             # 策略说明文档
└── optimization/         # 优化相关文件
```

## 更新日志

### v1.0.0
- 初始版本发布
- 实现完整的IVFF3策略逻辑
- 支持基本面筛选和分组选股
- 提供详细的参数配置选项
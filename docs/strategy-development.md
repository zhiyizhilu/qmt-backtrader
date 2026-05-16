# 策略开发与优化

## 创建自定义策略

1. 在 `strategies` 目录下创建新的策略子目录（推荐）或策略文件
2. 继承 `core.strategy_logic.StrategyLogic` 类（单标的策略）或 `core.stock_selection.StockSelectionStrategy` 类（多标的选股策略）
3. 使用 `@register_strategy` 装饰器注册策略
4. 实现相应的策略逻辑方法

> **注意**：策略文件放入 `strategies/` 目录后会自动被发现和注册，无需手动导入。

## 策略目录结构（推荐）

```
strategies/my_strategy/
├── __init__.py                  # 导出策略类
├── my_strategy.py               # 策略主文件
├── readme.md                    # 策略文档
├── backtest_results/            # 回测结果记录
└── optimization/                # 优化案例
    ├── run_optimization.py      # 自动化回测脚本
    ├── generate_report.py       # HTML 报告生成
    ├── plot_comparison.py       # 对比图生成
    ├── 优化报告.md               # 优化报告文档
    └── optimization_results/    # 优化回测结果
```

## 示例策略

### 单标的策略示例

```python
from core.strategy_logic import StrategyLogic, BarData
from strategies import register_strategy

@register_strategy('my_strategy', default_kwargs={'fast_period': 5, 'slow_period': 20},
                   backtest_config={'cash': 100000, 'commission': 0.0001,
                                    'start_date': '2025-01-01', 'end_date': '2026-04-17'})
class MyStrategy(StrategyLogic):
    params = (
        ('fast_period', 5),
        ('slow_period', 20),
        ('symbol', '000001.SZ'),
        ('position_ratio', 0.9),
    )

    def on_bar(self, bar: BarData):
        symbol = self.params.symbol
        close_prices = self.get_close_prices(symbol)

        if len(close_prices) < self.params.slow_period:
            return

        fast_ma = sum(close_prices[-self.params.fast_period:]) / self.params.fast_period
        slow_ma = sum(close_prices[-self.params.slow_period:]) / self.params.slow_period

        pos_size = self.get_position_size(symbol)

        if fast_ma > slow_ma and pos_size == 0:
            price = self.get_current_price(symbol)
            cash = self.get_cash()
            volume = int(cash * self.params.position_ratio / price / 100) * 100
            if volume >= 100:
                self.buy(symbol, price, volume)
        elif fast_ma < slow_ma and pos_size > 0:
            price = self.get_current_price(symbol)
            self.sell(symbol, price, pos_size)
```

### 选股策略示例

```python
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy

@register_strategy('my_selection_strategy', default_kwargs={'max_stocks': 10},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2016-01-01', 'end_date': '2026-04-17'})
class MySelectionStrategy(StockSelectionStrategy):
    """自定义选股策略"""

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
    )

    def select_stocks(self):
        """选股逻辑"""
        pool = self.get_stock_pool()
        filtered = self.screen_stocks(
            lambda s: (self.get_financial_field(s, 'Pershareindex', 'roe_diluted') or 0) > 0.10,
            pool
        )
        ranked = self.rank_stocks(
            lambda s: self.get_financial_field(s, 'Pershareindex', 'roe_diluted') or 0,
            stock_pool=filtered,
            top_n=self.params.max_stocks
        )
        return [stock for stock, _ in ranked]
```

## 内置策略

### 1. 高股息策略 (high_dividend)
- 基于股息率选股，行业分散配置
- 规避高股息陷阱，要求 ROE>0、归母净利润增速>0、经营现金流>0
- 支持近3年平均股息率模式，提升稳定性
- 双周调仓（`biweekly`），等权重持仓
- **防未来数据**：财务数据按公告日期索引，回测时只使用已披露数据
- 优化记录：波动率过滤、止损机制、ROE阈值、现金流阈值、双周调仓等

### 2. 小市值策略 (small_cap)
- 基本面过滤 + 行业分散 + 动量确认 + 波动率过滤 + 止损机制的小市值选股策略
- 四重基本面过滤：ROE>0、营收增速>0、经营现金流>0、资产负债率<阈值
- 每个申万一级行业选市值最小的1只，避免行业集中暴露
- 动量确认：近N日涨幅>0，避免在下跌趋势中接飞刀
- 波动率过滤：剔除日波动率过高的投机标的（事前风控）
- 止损机制：持仓亏损>=8%时在调仓时止损卖出（事后风控）
- 月度调仓，等权重持仓
- 优化记录：波动率过滤（夏普+14.5%）、止损机制（夏普+12.8%）、组合优化（夏普+22.5%）

## 策略优化工作流

框架内置了系统化的策略优化工作流，支持从提出优化建议到验证改进效果的全流程。

### 优化流程

1. **理解策略**：读取策略代码和文档，运行基线回测，确定核心评估指标
2. **提出优化建议**：提出10项具体优化建议，每项包含方向、实现路径、预期目标
3. **独立实施并回测**：每项优化独立实施（新参数默认禁用），运行回测，记录结果
4. **评估与筛选**：夏普比率相对基线提升 >= 5% 的优化予以保留
5. **组合有效优化**：测试所有有效优化的组合效果，检测参数冲突
6. **清理与文档**：删除无效优化代码，更新策略文档，生成 HTML 优化报告

### 已验证的优化经验

| 优化方案 | 适用策略类型 | 典型影响 | 说明 |
|---------|------------|---------|------|
| 波动率过滤 | 选股策略 | 夏普+10~15% | 过滤日波动率超过阈值的标的，首选优化方向 |
| 止损机制 | 选股策略 | 夏普+10~15% | 调仓时剔除亏损超过阈值的股票 |
| 波动率+止损组合 | 选股策略 | 夏普+20~25% | 两项优化互补叠加 |

### 优化输出目录

```
strategies/<策略目录>/optimization/
├── run_optimization.py          # 自动化回测运行脚本
├── generate_report.py           # HTML 报告生成脚本
├── plot_comparison.py           # 对比图生成脚本
├── 优化报告.md                   # 优化报告文档
├── optimization_report.html     # HTML 优化报告
└── optimization_results/        # 优化回测结果
    ├── baseline.json            # 基线结果
    ├── opt01_xxx.json           # 优化1结果
    └── ...
```

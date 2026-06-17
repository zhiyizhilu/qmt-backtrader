# 策略开发与优化

## 创建自定义策略

1. 在 `strategies` 目录下创建新的策略子目录（推荐）或策略文件
2. 继承 `core.strategy_logic.StrategyLogic` 类（单标的策略）或 `core.stock_selection.StockSelectionStrategy` 类（多标的选股策略）
3. 使用 `@register_strategy` 装饰器注册策略
4. 实现相应的策略逻辑方法

> **注意**：策略文件放入 `strategies/` 目录后会自动被发现和注册，无需手动导入。框架也会自动发现 `strategies_for_vip/` 和 `strategies_for_svip/` 目录下的策略。

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
from core.weight_allocator import RiskParityAllocator
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

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, weight_allocator=RiskParityAllocator(lookback=60), **kwargs)

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

## 使用不复权数据回测

框架默认使用后复权（`hfq`）数据进行回测，适用于大多数场景。某些策略（如打板策略、涨停检测、资金流向分析等）需要使用不复权数据，因为后复权价格会因除权除息而失真，导致涨停判断错误。

### 启用方式

在策略的 `@register_strategy` 装饰器中，通过 `default_kwargs` 传入 `dividend_type='none'` 即可启用不复权数据模式：

```python
@register_strategy('my_strategy',
                   default_kwargs={'max_stocks': 20, 'dividend_type': 'none'},
                   backtest_config={'cash': 1000000, 'commission': 0.0003,
                                    'start_date': '2025-06-01', 'end_date': '2026-06-01'})
class MyStrategy(StockSelectionStrategy):
    ...
```

### 数据流说明

```
策略注册 dividend_type='none'
    ↓
add_strategy() 从 kwargs 提取 dividend_type
    ↓
add_data(dividend_type='none')
    ↓
QMTDataProcessor.get_raw_data()    ← 不复权数据
    ↓
@smart_cache(namespace_suffix='_Raw')
    ↓
缓存到 .cache/QMTData/market_raw/{symbol}/{year}_{period}.parquet
```

- **后复权模式**（默认）：`add_data()` → `get_data()` → 缓存到 `market/` 目录
- **不复权模式**：`add_data(dividend_type='none')` → `get_raw_data()` → 缓存到 `market_raw/` 目录

两种模式的缓存完全独立，互不干扰。

### 数据预下载

不复权数据量较大（尤其是分钟线），建议在回测前预先下载：

```bash
# 下载不复权日线数据
python download_qmt_market_data.py --pool 中证1000 --type market_raw --period 1d

# 下载不复权分钟线数据
python download_qmt_market_data.py --pool 中证1000 --type market_raw --period 1m
```

预下载的数据会缓存到 `market_raw` 目录，回测时 `smart_cache` 直接读取本地缓存，不会重复下载。

### 适用场景

| 场景 | 说明 | 推荐复权方式 |
|------|------|-------------|
| 涨停/跌停检测 | 后复权价格可能超过涨跌停价，导致误判 | 不复权 |
| 打板策略 | 需要判断实际价格是否触及涨停价 | 不复权 |
| 资金流向分析 | 需要实际成交价格和金额 | 不复权 |
| 股息率计算 | 需要实际价格计算收益率 | 不复权 |
| 一般趋势跟踪 | 后复权价格连续，适合技术分析 | 后复权（默认） |
| 均线/动量策略 | 后复权消除除权缺口，信号更准确 | 后复权（默认） |

### 注意事项

1. **不指定 `dividend_type` 时默认使用后复权数据**，已有策略无需任何修改
2. 不复权数据在除权除息日会出现价格跳空，均线等技术指标可能失真
3. 不复权模式的缓存与后复权模式完全隔离，不会互相影响
4. 分钟级不复权数据量较大，建议通过 `download_qmt_market_data.py` 预下载

## 内置策略

### 标准策略（strategies/）

#### 1. 小市值策略 (small_cap)
- 基本面过滤 + 行业分散 + 动量确认 + 波动率过滤 + 止损机制的小市值选股策略
- 四重基本面过滤：ROE>0、营收增速>0、经营现金流>0、资产负债率<阈值
- 每个申万一级行业选市值最小的1只，避免行业集中暴露
- 动量确认：近N日涨幅>0，避免在下跌趋势中接飞刀
- 波动率过滤：剔除日波动率过高的投机标的（事前风控）
- 止损机制：持仓亏损>=8%时在调仓时止损卖出（事后风控）
- 月度调仓，等权重持仓
- 优化记录：波动率过滤（夏普+14.5%）、止损机制（夏普+12.8%）、组合优化（夏普+22.5%）

#### 2. 特质波动率因子策略 (ivff3)
- 基于 Fama-French 三因子模型的特质波动率选股策略，复现东方证券研报
- 构建 MKT、SMB、HML 三因子，对每只股票回归得到残差
- 残差的年化标准差即为特质波动率 IVFF3
- 选择 IVFF3 最低的股票构建等权组合（特质波动率之谜）
- 支持基本面筛选（ROE、净利润增长率、资产负债率）
- 月度调仓，最多50只股票

#### 3. 低估价值策略 (undervalued)
- 基于迈克尔·普莱斯与本杰明·格雷厄姆的价值选股法
- PB < 1.8（严格低估）、资产负债率 > 市场均值（逆向指标）、流动比率 >= 1.2
- 20日动量过滤：跌幅不超过-8%
- 月度调仓（优化后，原季度调仓），等权重持仓
- 优化记录：月度调仓（夏普+10.2%），样本外验证通过

#### 4. 聚宽小市值策略 (jq_small_cap)
- 克隆自聚宽文章，按流通市值升序选股
- 每日调仓，等权持仓，最多5只股票
- 股票池：中小综指
- 回测年化收益率 40.85%，夏普比率 1.4135

#### 5. 银行轮动策略 (bank_rotation)
- 四大银行（工、农、建、中）动量轮动策略
- 计算四只银行股的涨跌幅比率，买入当日相对跌幅最大的银行股
- 利用银行股之间的均值回归效应
- 支持日线和分钟线两种模式
- 经历多轮系统性优化（滑点优化、参数扰动测试、Walk-Forward 验证）

#### 6. 高股息低市盈率高增长价投策略 (dividend_value_growth)
- 克隆自聚宽文章，主打稳健：高分红、低市盈率、高增长
- 按近3年股息率排序，选取前10%的股票
- 基本面筛选：PE 0~25，PEG 0.08~1.9，ROE>3%，营收增速>5%，净利润增速>11%
- 月度调仓，等权重持仓，最多10只股票
- 回测年化收益率 24.74%，夏普比率 1.18，最大回撤 -17.82%

#### 7. 医药行业多因子选股策略 (medical_multi_factor)
- 基于 rank IC 赋权的多因子模型，复现聚宽社区策略
- 因子池：PB值、市值对数、换手率、ROE
- 因子权重：学习周期内各因子的 rank IC 均值
- 数据预处理：中位数去极值（MAD）+ Z-score 标准化
- 月度调仓，等权持仓，最多20只股票

#### 8. 二八轮动小市值策略 (twenty_eight_rotation)
- 基于沪深300和中证500指数20日涨幅择时，小市值选股
- 二八止损：两个指数20日涨幅都为负时清仓避险
- 可选大盘止损和个股止损机制
- 按市值升序排列，选取前N只小市值股票
- 每5个交易日调仓，等权重持仓

#### 9. 首板低开策略 (first_board_low_open)
- 选取"低位、非连板、涨停后次日低开"的股票
- 筛选昨日涨停、排除连板、60日内相对位置<=50%
- 筛选今日低开3%-4%的股票，开盘买入
- 次日上午有盈利就卖出，没有就拿到尾盘
- 支持日线和分钟线两种模式

#### 10. 罗斯曼价值精选策略 (rothman_value)
- 源自华尔街首席分析师 Howard Rothman 的审慎致富投资法
- 六大选股条件：总市值≥200亿、流动比率≥1.0、ROE≥8%、每股经营现金流>0、净利润增速≥0%、盈余报酬率≥2%
- 综合评分体系：ROE(35%) + 利润增速(25%) + 盈余报酬率(25%) + 现金流(15%)
- 月度调仓，等权重持仓，最多20只股票

## 策略优化工作流

框架内置了系统化的策略优化工作流，支持从提出优化建议到验证改进效果的全流程。

### 优化流程

1. **理解策略**：读取策略代码和文档，运行基线回测，确定核心评估指标
2. **提出优化建议**：提出10项具体优化建议，每项包含方向、实现路径、预期目标
3. **独立实施并回测**：每项优化独立实施（新参数默认禁用），运行回测，记录结果
4. **评估与筛选**：夏普比率相对基线提升 >= 5% 的优化予以保留
5. **组合有效优化**：测试所有有效优化的组合效果，检测参数冲突
6. **清理与文档**：删除无效优化代码，更新策略文档，生成 HTML 优化报告

### Walk-Forward 样本外验证

使用 `WalkForwardSplitter` 进行样本外验证，防止过拟合：

```python
from utils.walk_forward import WalkForwardSplitter

splitter = WalkForwardSplitter(
    total_start='2015-01-01',
    total_end='2026-04-28',
    train_months=60,
    test_months=12,
    step_months=12,
    anchor=False,
)

for train_start, train_end, test_start, test_end in splitter.split():
    print(f"训练: {train_start}~{train_end}, 测试: {test_start}~{test_end}")
```

- **滚动模式**（anchor=False）：训练窗口固定长度向前滑动
- **锚定模式**（anchor=True）：训练窗口起点固定，终点向前扩展

### 已验证的优化经验

| 优化方案 | 适用策略类型 | 典型影响 | 说明 |
|---------|------------|---------|------|
| 波动率过滤 | 选股策略 | 夏普+10~15% | 过滤日波动率超过阈值的标的，首选优化方向 |
| 止损机制 | 选股策略 | 夏普+10~15% | 调仓时剔除亏损超过阈值的股票 |
| 波动率+止损组合 | 选股策略 | 夏普+20~25% | 两项优化互补叠加 |
| 月度调仓 | 价值策略 | 夏普+10% | 价值回归需要时间，月度是"捕捉机会"与"耐心等待"的平衡点 |

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

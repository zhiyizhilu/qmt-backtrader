# 量化交易框架

一个基于 Backtrader 和 QMT 的量化交易框架。**核心目标：一套策略代码无缝切换回测、模拟交易和实盘交易**，无需为不同运行模式修改策略逻辑，极大降低从研究到上线的迁移成本。

## 功能特点

- **多模式支持**：回测、模拟交易、实盘交易
- **高性能回测**：基于 Backtrader 引擎
- **QMT 集成**：支持通过 QMT 进行模拟和实盘交易
- **多数据源**：支持 QMT、AkShare、Baostock 等数据源
- **智能缓存**：内存 + 磁盘双层缓存，支持 parquet 格式，中断后自动恢复
- **策略模板**：内置双均线策略、高股息策略、ETF 轮动策略等
- **参数优化**：支持网格搜索和遗传算法
- **可视化**：提供回测结果和实时监控的可视化
- **机器学习集成**：支持机器学习模型的训练和预测
- **选股策略**：支持基于基本面的股票选择策略

## 项目结构

```
qmt_backtrader/
├── README.md                    # 项目说明文档
├── main.py                      # 主入口文件
├── requirements.txt             # 依赖包管理
├── .cache/                      # 数据缓存目录（自动创建）
├── api/                         # API 接口
│   ├── __init__.py
│   ├── backtest_api.py          # 回测 API（基于 Backtrader）
│   ├── base_api.py              # API 基类
│   └── qmt_api.py               # QMT 交易 API
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── analyzer.py              # 分析器
│   ├── cache.py                 # 智能缓存系统（内存+磁盘+增量合并）
│   ├── data.py                  # 数据处理（QMT/AkShare/Baostock）
│   ├── data_adapter.py          # 数据适配器
│   ├── executor.py              # 执行器
│   ├── financial_data.py        # 财务数据处理
│   ├── models.py                # 数据模型
│   ├── stock_selection.py       # 选股策略基类
│   ├── strategy.py              # 策略基类
│   └── strategy_logic.py        # 策略逻辑
├── example/                     # 示例文档
├── logs/                        # 日志目录
├── monitor/                     # 监控模块
│   ├── __init__.py
│   └── realtime_monitor.py      # 实时监控
├── strategies/                  # 策略目录
│   ├── __init__.py
│   ├── config.py                # 策略配置
│   ├── etf_rotation_strategy.py # ETF 轮动策略
│   ├── example_strategy.py      # 示例策略
│   ├── fundamental_strategy.py  # 基本面策略
│   └── high_dividend_strategy.py # 高股息策略
└── utils/                       # 工具函数
    ├── __init__.py
    ├── logger.py                # 日志管理
    ├── parameter_optimizer.py   # 参数优化工具
    ├── report.py                # 报告生成
    └── visualization.py         # 可视化工具
```

## 环境搭建

1. **安装 Python**：建议使用 Python 3.9+

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **安装 QMT**（可选，用于实盘/模拟交易）：
   - 从券商获取 QMT 客户端
   - 安装并登录 QMT
   - 勾选"独立交易"进入 MiniQMT 模式
   - 从 QMT 官网下载并安装 XtQuant 库

## 使用方法

### 1. 运行回测

```bash
python main.py --mode backtest --strategy high_dividend --period 1d --pool 沪深300 --start 2025-04-21 --end 2026-04-21 --data-source baostock
```

**参数说明**：
- `--mode`：运行模式，可选值：`backtest`（回测）、`sim`（模拟交易）、`real`（实盘交易）
- `--strategy`：策略类型，可选值：`double_ma`、`high_dividend`、`etf_rotation` 等
- `--period`：数据周期，可选值：`1d`（日线）、`1m`、`5m`、`15m`、`30m`、`60m`、`tick`
- `--pool`：股票池板块名称，如 `沪深300`、`沪深A股`、`上证50`
- `--start`：回测起始日期，格式：`YYYY-MM-DD`
- `--end`：回测结束日期，格式：`YYYY-MM-DD`
- `--data-source`：数据源，可选值：`qmt`（需 QMT 客户端）、`akshare`（免费在线）、`baostock`（免费在线）

### 2. 运行模拟交易

```bash
python main.py --mode sim --strategy double_ma
```

### 3. 运行实盘交易

```bash
python main.py --mode real --strategy double_ma
```

## 智能缓存系统

框架内置了双层智能缓存系统，大幅提升数据加载速度：

### 缓存架构

- **内存缓存（MemCache）**：基于 OrderedDict 的线程安全 LRU 缓存，默认容量 500
- **磁盘缓存（DiskCache）**：支持 parquet 和 pickle 两种格式，按命名空间隔离

### 缓存特性

1. **逐只缓存**：财务数据按股票代码单独缓存，中断后自动恢复，避免重复下载
2. **合并缓存**：全部下载完成后自动合并为总缓存，提升后续加载速度
3. **增量更新**：行情数据支持智能增量更新，只下载缺失部分
4. **格式优化**：默认使用 pyarrow parquet 格式，比 pickle 更快更紧凑
5. **原子写入**：使用临时文件 + 原子重命名，避免写入中断导致文件损坏

### 缓存目录

缓存文件默认存储在项目根目录的 `.cache/` 文件夹下：
```
.cache/
├── BaoStockDataProcessor/           # Baostock 行情数据缓存
├── BaoStockDataProcessor_Financial/ # Baostock 财务数据缓存
│   ├── 600000.SH_2024Q1_2026Q1.parquet
│   └── merged_300_2025-04-21_2026-04-21.parquet
└── AKShareDataProcessor/            # AkShare 数据缓存
```

### 环境变量

- `QMT_CACHE_DIR`：自定义缓存目录路径
- `QMT_MEM_CACHE_LIMIT`：内存缓存容量限制（默认 500）

## 策略开发

### 创建自定义策略

1. 在 `strategies` 目录下创建新的策略文件
2. 继承 `core.strategy.BaseStrategy` 类（单标的策略）或 `core.stock_selection.StockSelectionStrategy` 类（多标的选股策略）
3. 使用 `@register_strategy` 装饰器注册策略
4. 实现相应的策略逻辑方法

### 示例策略

#### 单标的策略示例

```python
from core.strategy import BaseStrategy
import backtrader as bt
from strategies import register_strategy

@register_strategy('my_strategy', default_kwargs={'fast_period': 5, 'slow_period': 20},
                   backtest_config={'cash': 100000, 'commission': 0.0001,
                                    'start_date': '2025-01-01', 'end_date': '2026-04-17'})
class MyStrategy(BaseStrategy):
    params = (
        ('fast_period', 5),
        ('slow_period', 20),
    )
    
    def __init__(self):
        super().__init__()
        # 初始化指标
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.slow_period)
    
    def next(self):
        # 策略逻辑
        if self.fast_ma[0] > self.slow_ma[0] and self.fast_ma[-1] <= self.slow_ma[-1]:
            if not self.position:
                self.buy()
        elif self.fast_ma[0] < self.slow_ma[0] and self.fast_ma[-1] >= self.slow_ma[-1]:
            if self.position:
                self.sell()
```

#### 选股策略示例

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
        # 实现自定义选股逻辑
        # ...
        return selected_stocks
```

## 内置策略

### 1. 双均线策略 (double_ma)
- 基于短期和长期移动平均线的交叉信号
- 适用于趋势跟踪

### 2. 高股息策略 (high_dividend)
- 基于股息率选股，行业分散配置
- 规避高股息陷阱，要求 ROE>0、净利润增速>0、经营现金流>0
- 月度调仓，等权重持仓
- **防未来数据**：财务数据按公告日期索引，回测时只使用已披露数据

### 3. ETF 轮动策略 (etf_rotation)
- 基于 ETF 的动量和波动率进行轮动
- 选择表现较好的 ETF 进行配置

### 4. 基本面策略 (fundamental)
- 基于基本面指标选股
- 可自定义多种财务指标过滤条件

## 数据源对比

| 数据源 | 实时性 | 数据范围 | 使用门槛 | 推荐场景 |
|--------|--------|----------|----------|----------|
| QMT | 实时 | A股全市场 | 需开户+客户端 | 实盘/模拟交易 |
| AkShare | 延迟 | A股/期货/外汇等 | 免费，无需账户 | 研究/回测 |
| Baostock | 延迟 | A股历史数据 | 免费，无需账户 | 历史回测 |

## 注意事项

1. **实盘交易风险**：实盘交易存在风险，请谨慎使用
2. **QMT 依赖**：使用 QMT 接口需要安装 QMT 客户端和 XtQuant 库
3. **数据质量**：回测结果依赖于数据质量，请确保数据的准确性
4. **策略优化**：参数优化可能导致过拟合，请谨慎使用优化结果
5. **数据源选择**：不同数据源的质量和覆盖范围不同，请根据需要选择合适的数据源
6. **缓存清理**：如遇到数据异常，可手动删除 `.cache/` 目录下的对应文件重新下载

## 后续计划

- [ ] 支持更多交易接口（如tushare、yahoo等）
- [ ] 实现策略自动优化
- [ ] 增加深度学习模型集成
- [ ] 开发 Web 界面
- [ ] 实现分布式回测和交易
- [ ] 增加更多因子和策略模板
- [ ] 优化回测性能
- [ ] 完善风险管理功能
- [ ] 添加实时告警系统

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用者应自行承担使用本项目的全部风险和责任。

## 风险提示

1. **市场风险**：金融市场存在不可预测的风险，过往表现不代表未来收益
2. **技术风险**：自动化交易系统可能存在技术故障、网络延迟等问题
3. **合规风险**：请确保您的交易行为符合当地法律法规要求
4. **数据风险**：免费数据源可能存在延迟或错误，实盘前请验证数据准确性
5. **资金风险**：实盘交易可能导致资金损失，请谨慎投资

**投资有风险，入市需谨慎！**

## 联系作者

如果您对项目有任何问题或建议，欢迎添加作者微信交流：

![微信二维码](https://mmbiz.qpic.cn/sz_mmbiz_jpg/bibrY3tFKMBH9VM46EcxmAIG1rwQR3ZEibueW40mJb38Fib4RU7AzrSqfYxLH0pAibHiaYUxbwQtQrv1iaticN7JmRRiaQ/640?wx_fmt=other&from=appmsg#imgIndex=7)

扫描上方二维码添加好友，获取更多项目更新和技术支持。

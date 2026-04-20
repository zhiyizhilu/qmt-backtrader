# 量化交易框架

一个基于Backtrader和QMT的量化交易框架。**核心目标：一套策略代码无缝切换回测、模拟交易和实盘交易**，无需为不同运行模式修改策略逻辑，极大降低从研究到上线的迁移成本。

## 功能特点

- **多模式支持**：回测、模拟交易、实盘交易
- **高性能回测**：基于Backtrader引擎
- **QMT集成**：支持通过QMT进行模拟和实盘交易
- **多数据源**：支持QMT、AkShare、Baostock等数据源
- **策略模板**：内置双均线策略、高股息策略、ETF轮动策略等
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
├── api/                         # API接口
│   ├── __init__.py
│   ├── backtest_api.py          # 回测API (基于Backtrader)
│   ├── base_api.py              # API基类
│   └── qmt_api.py               # QMT交易API
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── analyzer.py              # 分析器
│   ├── data.py                  # 数据处理
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
│   ├── etf_rotation_strategy.py # ETF轮动策略
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

1. **安装Python**：建议使用Python 3.9+

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **安装QMT**：
   - 从券商获取QMT客户端
   - 安装并登录QMT
   - 勾选"独立交易"进入MiniQMT模式
   - 从QMT官网下载并安装XtQuant库

## 使用方法

### 1. 运行回测

```bash
python main.py --mode backtest --strategy double_ma --period 1d --pool 沪深A股 --start 2016-01-01 --end 2026-04-17 --data-source qmt
```

**参数说明**：
- `--mode`：运行模式，可选值：backtest（回测）、sim（模拟交易）、real（实盘交易）
- `--strategy`：策略类型，可选值：double_ma、high_dividend、etf_rotation等
- `--period`：数据周期，可选值：1d（日线）、1m（1分钟）、5m（5分钟）、15m（15分钟）、30m（30分钟）、60m（60分钟）、tick（tick数据）
- `--pool`：股票池板块名称，如"沪深A股"
- `--start`：回测起始日期，格式：YYYY-MM-DD
- `--end`：回测结束日期，格式：YYYY-MM-DD
- `--data-source`：数据源，可选值：qmt（需QMT客户端）、akshare（免费在线）、baostock（免费在线）

### 2. 运行模拟交易

```bash
python main.py --mode sim --strategy double_ma
```

### 3. 运行实盘交易

```bash
python main.py --mode real --strategy double_ma
```

## 策略开发

### 创建自定义策略

1. 在`strategies`目录下创建新的策略文件
2. 继承`core.strategy.BaseStrategy`类（单标的策略）或`core.stock_selection.StockSelectionStrategy`类（多标的选股策略）
3. 使用`@register_strategy`装饰器注册策略
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
- 规避高股息陷阱，要求ROE>0、净利润增速>0、经营现金流>0
- 月度调仓，等权重持仓

### 3. ETF轮动策略 (etf_rotation)
- 基于ETF的动量和波动率进行轮动
- 选择表现较好的ETF进行配置

### 4. 基本面策略 (fundamental)
- 基于基本面指标选股
- 可自定义多种财务指标过滤条件

## 注意事项

1. **实盘交易风险**：实盘交易存在风险，请谨慎使用
2. **QMT依赖**：使用QMT接口需要安装QMT客户端和XtQuant库
3. **数据质量**：回测结果依赖于数据质量，请确保数据的准确性
4. **策略优化**：参数优化可能导致过拟合，请谨慎使用优化结果
5. **数据源选择**：不同数据源的质量和覆盖范围不同，请根据需要选择合适的数据源

## 后续计划

- 支持更多交易接口
- 实现策略自动优化
- 增加深度学习模型集成
- 开发Web界面
- 支持更多资产类别
- 实现分布式回测和交易
- 增加更多因子和策略模板
- 优化回测性能
- 完善风险管理功能




---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用者应自行承担使用本项目的全部风险和责任。

## 风险提示

1. **市场风险**：金融市场存在不可预测的风险，过往表现不代表未来收益
2. **技术风险**：自动化交易系统可能存在技术故障、网络延迟等问题
3. **合规风险**：请确保您的交易行为符合当地法律法规要求
4. **资金风险**：实盘交易可能导致资金损失，请谨慎投资

**投资有风险，入市需谨慎！**

## 联系作者

如果您对项目有任何问题或建议，欢迎添加作者微信交流：

![微信二维码](https://mmbiz.qpic.cn/sz_mmbiz_jpg/bibrY3tFKMBH9VM46EcxmAIG1rwQR3ZEibueW40mJb38Fib4RU7AzrSqfYxLH0pAibHiaYUxbwQtQrv1iaticN7JmRRiaQ/640?wx_fmt=other&from=appmsg#imgIndex=7)

扫描上方二维码添加好友，获取更多项目更新和技术支持。

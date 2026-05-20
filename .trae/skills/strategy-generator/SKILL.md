---
name: "strategy-generator"
description: "从策略文档（Markdown/PDF/Word/URL）自动理解策略逻辑，在本项目中生成可运行的量化策略代码，并完成回测。当用户提供策略文档、URL链接或策略描述，要求实现量化策略时调用此技能。"
---

# 策略生成器

从策略文档自动理解策略逻辑，在 qmt_backtrader 框架中生成可运行的量化策略代码，并完成回测。

## 环境配置

**Python 运行环境**（通过 PowerShell 动态获取系统 Python 环境）：

```powershell
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = (Get-ChildItem "$env:USERPROFILE\AppData\Local\Programs\Python\Python*" -Directory |
        Sort-Object Name -Descending | Select-Object -First 1).FullName + "\python.exe"
}
```

- 所有回测、脚本、pip 安装等 Python 操作必须使用动态获取的 Python 路径
- 示例：`& "$pythonPath" main.py --mode backtest ...`
- 如需指定完整路径而非使用 PATH 中的 Python，可通过 `sys.executable` 或上述脚本获取后调用
- 项目根目录：`e:\jupyter notebook\automatic\qmt_backtrader`

## 调用时机

当用户出现以下情况时调用此技能：
- 提供策略文档路径（Markdown/PDF/Word），要求实现策略
- 提供 URL 链接，要求根据网页内容实现策略
- 直接粘贴策略描述文本，要求实现
- 说"根据这个文档写策略"、"实现这个策略"、"帮我回测这个策略"等

## 核心工作流

### 阶段一：文档读取与策略理解

#### 1.1 输入处理

根据输入类型选择读取方式：

| 输入类型 | 识别方式 | 处理方法 |
|---------|---------|---------|
| 本地 Markdown (`.md`) | 路径以 `.md` 结尾 | 使用 Read 工具直接读取 |
| 本地 PDF (`.pdf`) | 路径以 `.pdf` 结尾 | 使用 Python 脚本提取文本（见下方"PDF提取脚本"） |
| 本地 Word (`.docx`/`.doc`) | 路径以 `.docx`/`.doc` 结尾 | 使用 Python 脚本提取文本（见下方"Word提取脚本"） |
| 本地纯文本 (`.txt`) | 路径以 `.txt` 结尾 | 使用 Read 工具直接读取 |
| URL 链接 | 以 `http://` 或 `https://` 开头 | 使用 WebFetch 工具抓取，自动转为 markdown |
| 直接粘贴文本 | 无文件路径特征 | 直接使用文本内容 |

**PDF 提取脚本**：

```python
import sys
try:
    import pdfplumber
    with pdfplumber.open(sys.argv[1]) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                print(text)
except ImportError:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(sys.argv[1])
        for page in reader.pages:
            text = page.extract_text()
            if text:
                print(text)
    except ImportError:
        print("ERROR: 需要安装 pdfplumber 或 PyPDF2: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)
```

运行方式：使用 `$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"; & "$pythonPath" -c "<上述脚本>" <pdf文件路径>`

**Word 提取脚本**：

```python
import sys
try:
    from docx import Document
    doc = Document(sys.argv[1])
    for para in doc.paragraphs:
        if para.text.strip():
            print(para.text)
except ImportError:
    print("ERROR: 需要安装 python-docx: pip install python-docx", file=sys.stderr)
    sys.exit(1)
```

运行方式：使用 `$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"; & "$pythonPath" -c "<上述脚本>" <docx文件路径>`

#### 1.2 策略要素提取

从文档内容中提取以下关键策略要素：

| 要素 | 说明 | 提取方法 |
|------|------|---------|
| 策略类型 | 选股调仓型 vs 择时/轮动型 | 根据关键词判断（见1.3） |
| 选股/交易逻辑 | 买卖条件、指标计算、信号生成规则 | 识别条件语句、公式、规则描述 |
| 参数定义 | 可调参数及其默认值 | 识别数值参数、阈值、周期等 |
| 数据需求 | 需要哪些行情数据、财务数据 | 识别指标名称、数据字段 |
| 调仓规则 | 调仓频率、仓位管理方式 | 识别"月度/周/季度调仓"、"等权"等 |
| 风控规则 | 止损、止盈、仓位限制 | 识别"止损X%"、"最大回撤"等 |
| 回测配置 | 资金、日期范围、股票池 | 识别回测区间、初始资金等 |

#### 1.3 策略类型自动判断

| 文档特征 | 判断为 | 基类 |
|---------|--------|------|
| 提到"选股"、"股票池"、"调仓"、"持仓组合"、"等权" | 选股调仓型 | `StockSelectionStrategy` |
| 提到"轮动"、"择时"、"均线交叉"、"买卖信号"、"金叉死叉" | 择时/轮动型 | `StrategyLogic` |
| 提到"ETF轮动"、"标的切换"、"动量轮动" | 择时/轮动型 | `StrategyLogic` |
| 同时包含选股和择时特征 | 优先选股调仓型 | `StockSelectionStrategy` |
| 无法明确判断 | 询问用户选择 | - |

### 阶段二：策略代码生成

#### 2.1 策略命名规则

根据策略核心逻辑生成有意义的英文注册名，参考现有策略命名风格：

| 策略逻辑 | 推荐命名 | 示例 |
|---------|---------|------|
| 高股息选股 | `high_dividend` | 已有 |
| 小市值选股 | `small_cap` | 已有 |
| ETF动量轮动 | `etf_rotation` | 已有 |
| 低波动率选股 | `low_volatility` | - |
| 质量因子选股 | `quality_factor` | - |
| 价值成长选股 | `value_growth` | - |
| 双均线择时 | `double_ma` | 已有 |
| 多因子选股 | `multi_factor` | - |
| 行业轮动 | `sector_rotation` | - |
| 动量策略 | `momentum` | - |

**命名原则**：
- 使用小写英文 + 下划线
- 2-3个词，简洁明确
- 体现策略核心逻辑而非具体参数
- 避免与已有策略重名（检查 `strategies/`、`strategies_for_vip/` 和 `strategies_my/` 目录）

#### 2.2 策略目录位置

| 场景 | 目录 |
|------|------|
| 默认 | `strategies_my/<strategy_name>_strategy/` |
| 用户指定放系统目录 | `strategies/<strategy_name>_strategy/` |
| 用户指定放 VIP 目录 | `strategies_for_vip/<strategy_name>_strategy/` |

#### 2.3 StockSelectionStrategy 代码模板（选股调仓型）

```python
from typing import List
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('<strategy_name>',
                   default_kwargs={'max_stocks': <N>},
                   backtest_config={'cash': 1000000, 'commission': 0.0001,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '<股票池>'})
class <StrategyName>Strategy(StockSelectionStrategy):
    """<策略中文名> - <一句话描述>

    选股逻辑：
    1. <步骤1>
    2. <步骤2>
    3. <步骤3>

    调仓规则：
    - <调仓频率>调仓，等权重持仓
    """

    params = (
        ('rebalance_freq', '<freq>'),
        ('max_stocks', <N>),
        ('position_ratio', 0.95),
        ('stock_pool', None),
    )

    def select_stocks(self) -> List[str]:
        pool = self.get_stock_pool()

        # 步骤1: 基本面/技术面过滤
        filtered = self._filter_<filter_name>(pool)
        if not filtered:
            self.log(f'过滤后无股票')
            return []

        # 步骤2: 排序/评分
        scored = self._score_stocks(filtered)

        # 步骤3: 取前N只
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in scored[:max_stocks]]

        self.log(f'选股结果: {len(pool)} -> {len(filtered)} -> {len(selected)} 只')
        return selected

    def _filter_<filter_name>(self, pool: List[str]) -> List[str]:
        result = []
        for stock in pool:
            # 过滤逻辑
            pass
        return result

    def _score_stocks(self, stocks: List[str]) -> list:
        scored = []
        for stock in stocks:
            # 评分逻辑
            score = 0
            scored.append((stock, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
```

#### 2.4 StrategyLogic 代码模板（择时/轮动型）

```python
from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy


@register_strategy('<strategy_name>',
                   backtest_config={'cash': 200000, 'commission': 0.0005,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d'})
class <StrategyName>Strategy(StrategyLogic):
    """<策略中文名> - <一句话描述>

    交易逻辑：
    1. <条件1>时买入
    2. <条件2>时卖出
    """

    params = (
        ('<param1>', <default1>),
        ('<param2>', <default2>),
    )

    def on_bar(self, bar: BarData):
        # 获取数据
        # 判断信号
        # 执行交易
        pass

    def on_order(self, order: OrderInfo):
        super().on_order(order)

    def on_trade(self, trade: TradeInfo):
        super().on_trade(trade)
```

#### 2.5 输出目录结构

```
strategies_my/<strategy_name>_strategy/     (或 strategies/<strategy_name>_strategy/、strategies_for_vip/<strategy_name>_strategy/)
├── __init__.py                          # 空文件
├── <strategy_name>_strategy.py          # 策略主文件
└── readme.md                            # 策略说明文档
```

**readme.md 模板**：

```markdown
# <策略中文名>

## 策略概述

<策略的核心思想和目标>

## 选股/交易逻辑

1. <步骤1>
2. <步骤2>
3. <步骤3>

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rebalance_freq | monthly | 调仓频率 |
| max_stocks | 10 | 最大持仓数量 |

## 回测配置

- 初始资金: 1,000,000
- 股票池: 中证1000
- 回测区间: 2020-04-28 ~ 2026-04-28
- 佣金: 0.01%

## 数据来源

- <数据来源说明>
```

### 阶段三：回测配置与执行

#### 3.1 回测配置提取

从文档中提取回测配置，缺失项使用默认值：

| 配置项 | 选股策略默认值 | 择时策略默认值 | 提取来源 |
|--------|-------------|-------------|---------|
| 初始资金 (cash) | 1,000,000 | 200,000 | 文档中的"初始资金"、"资金量" |
| 佣金 (commission) | 0.0001 | 0.0005 | 文档中的"手续费"、"佣金" |
| 股票池 (pool) | 中证1000 | - | 文档中的"股票池"、"选股范围" |
| 回测起始 (start_date) | 2020-04-28 | 2020-04-28 | 文档中的"回测区间" |
| 回测结束 (end_date) | 2026-04-28 | 2026-04-28 | 文档中的"回测区间" |
| 基准 (benchmark) | 自动根据股票池 | 000300.SH | 文档中的"基准"、"对比指数" |

#### 3.2 回测执行

**命令行方式**：

```bash
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
& "$pythonPath" main.py --mode backtest --strategy <strategy_name> --period 1d --pool <股票池> --start <起始日期> --end <结束日期> --ai-mode --debug
```

**程序化方式**（推荐，可获取详细指标）：

StockSelectionStrategy 子类：

```python
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config
from core.data.index_constituent import IndexConstituentManager

strategy_name = '<strategy_name>'
strategy_class = get_strategy(strategy_name)
default_kwargs = get_strategy_default_kwargs(strategy_name)
backtest_config = get_strategy_backtest_config(strategy_name)

config = dict(backtest_config)
config['period'] = '1d'
benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(config.get('pool', ''), '000300.SH')
config.setdefault('benchmark', benchmark)

api = BacktestAPI()
api.set_ai_mode(True)
api.set_no_record(True)
api.configure(**config)
api.load_financial_data(sector=config.get('pool', '沪深A股'))
api.add_stock_selection_strategy(strategy_class, **default_kwargs)
results = api.run()

if results:
    result = api.get_result()
    sr = result.sharpe_ratio()
    dd = result.max_drawdown()
    acc = result.account
    print(f"夏普比率: {sr:.4f}")
    print(f"最大回撤: {dd*100:.2f}%")
    print(f"总收益率: {acc.rate*100:.2f}%")
    print(f"最终权益: {acc.dynamic_rights:.2f}")
else:
    print("回测未产生结果")
```

StrategyLogic 子类：

```python
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['QMT_LOG_LEVEL'] = 'WARNING'

from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config

strategy_name = '<strategy_name>'
strategy_class = get_strategy(strategy_name)
default_kwargs = get_strategy_default_kwargs(strategy_name)
backtest_config = get_strategy_backtest_config(strategy_name)

config = dict(backtest_config)
config['period'] = '1d'
config.setdefault('benchmark', '000300.SH')

api = BacktestAPI()
api.set_ai_mode(True)
api.set_no_record(True)
api.configure(**config)
api.add_strategy(strategy_class, **default_kwargs)
results = api.run()

if results:
    result = api.get_result()
    sr = result.sharpe_ratio()
    dd = result.max_drawdown()
    acc = result.account
    print(f"夏普比率: {sr:.4f}")
    print(f"最大回撤: {dd*100:.2f}%")
    print(f"总收益率: {acc.rate*100:.2f}%")
    print(f"最终权益: {acc.dynamic_rights:.2f}")
else:
    print("回测未产生结果")
```

**关键差异**：
- StockSelectionStrategy：`api.load_financial_data(sector=pool)` + `api.add_stock_selection_strategy()`
- StrategyLogic：`api.add_strategy()`（无 load_financial_data）
- 两种模板都应调用 `api.set_no_record(True)` 避免回测写入正式结果目录
- 两种模板都应调用 `api.set_ai_mode(True)` 跳过图形界面渲染

#### 3.3 回测结果报告

回测完成后，向用户报告以下核心指标：

| 指标 | 说明 |
|------|------|
| 夏普比率 | 风险调整后收益 |
| 总收益率 | 策略总回报 |
| 年化收益率 | 年化后的收益率 |
| 最大回撤 | 最大峰谷回撤 |
| 最终权益 | 回测结束时的账户权益 |

## 数据接口映射

策略文档中的自然语言描述需要映射到项目中已有的数据接口。

### 选股策略可用接口

| 文档常见描述 | 对应项目 API | 代码示例 |
|-------------|-------------|---------|
| 股息率、分红 | `self.get_dividend_yield(stock)` | `dy = self.get_dividend_yield('000001.SZ')` |
| ROE | `self.get_financial_field(stock, 'Pershareindex', 'du_return_on_equity')` | `roe = self.get_financial_field(s, 'Pershareindex', 'du_return_on_equity')` |
| EPS | `self.get_financial_field(stock, 'Pershareindex', 'eps_diluted')` | `eps = self.get_financial_field(s, 'Pershareindex', 'eps_diluted')` |
| 营收增速 | `self.get_financial_field(stock, 'Pershareindex', 'inc_net_profit_rate')` | `growth = self.get_financial_field(s, 'Pershareindex', 'inc_net_profit_rate')` |
| 每股经营现金流 | `self.get_financial_field(stock, 'Pershareindex', 's_fa_ocfps')` | `ocf = self.get_financial_field(s, 'Pershareindex', 's_fa_ocfps')` |
| 批量财务数据 | `self.get_financial_fields_batch(stocks, table, fields)` | `data = self.get_financial_fields_batch(pool, 'Pershareindex', ['eps_diluted', 'du_return_on_equity'])` |
| 行业分类 | `self.get_industry(stock)` | `industry = self.get_industry('000001.SZ')` |
| 同比增长率 | `self.compute_growth_rate(stock, table, field)` | `growth = self.compute_growth_rate(s, 'Income', 'total_operate_income')` |
| 股票筛选 | `self.screen_stocks(condition, pool)` | `selected = self.screen_stocks(lambda s: (self.get_financial_field(s, 'Pershareindex', 'eps_diluted') or 0) > 0.5)` |
| 排序选股 | `self.rank_stocks(score_func, pool, top_n)` | `ranked = self.rank_stocks(score_func, top_n=10)` |
| 当前价格 | `self.get_current_price(stock)` | `price = self.get_current_price('000001.SZ')` |
| 收盘价序列 | `self.get_close_prices(stock, period)` | `closes = self.get_close_prices('000001.SZ', 20)` |
| OHLCV数据 | `self.get_ohlcv_data(stock, period)` | `ohlcv = self.get_ohlcv_data('000001.SZ', 20)` |
| 股票池 | `self.get_stock_pool()` | `pool = self.get_stock_pool()` |
| 历史派息 | `self.get_dvps_history(stock, count)` | `dvps = self.get_dvps_history('000001.SZ', 3)` |
| 行业映射 | `self.get_industry_mapping()` | `mapping = self.get_industry_mapping()` |

### 择时策略可用接口

| 文档常见描述 | 对应项目 API | 代码示例 |
|-------------|-------------|---------|
| 收盘价序列 | `self.get_close_prices(symbol, period)` | `closes = self.get_close_prices(symbol, 20)` |
| 当前价格 | `self.get_current_price(symbol)` | `price = self.get_current_price(symbol)` |
| OHLCV数据 | `self.get_ohlcv_data(symbol, period)` | `ohlcv = self.get_ohlcv_data(symbol, 20)` |
| 持仓查询 | `self.get_position_size(symbol)` | `pos = self.get_position_size(symbol)` |
| 可用资金 | `self.get_cash()` | `cash = self.get_cash()` |
| 当前日期 | `self.get_current_date()` | `date = self.get_current_date()` |
| 买入 | `self.buy(symbol, price, volume)` | `self.buy(symbol, price, 100)` |
| 卖出 | `self.sell(symbol, price, volume)` | `self.sell(symbol, price, 100)` |
| 标的列表 | `self.get_symbols()` | `symbols = self.get_symbols()` |

### 财务数据表名与字段参考

| 表名 (table_name) | 常用字段 | 说明 |
|-------------------|---------|------|
| `Pershareindex` | `eps_diluted`, `du_return_on_equity`, `inc_net_profit_rate`, `s_fa_ocfps`, `bps` | 每股指标 |
| `Income` | `total_operate_income`, `operate_income`, `total_profit`, `net_profit` | 利润表 |
| `Balance` | `total_assets`, `total_liability`, `total_equity` | 资产负债表 |
| `Cashflow` | `net_operate_cash_flow`, `net_invest_cash_flow` | 现金流量表 |

**不确定字段名时**：查阅项目中的 `core/financial_data.py` 确认可用字段。

## 常见策略逻辑的代码实现参考

### 均线计算

```python
closes = self.get_close_prices(symbol, period + 1)
if len(closes) >= period:
    ma = sum(closes[-period:]) / period
```

### 波动率计算

```python
closes = self.get_close_prices(symbol, 21)
if len(closes) >= 2:
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
```

### RSI 计算

```python
closes = self.get_close_prices(symbol, period + 1)
if len(closes) >= period + 1:
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
```

### 行业分散选股

```python
from collections import defaultdict

industry_stocks = defaultdict(list)
for stock, score in scored_stocks:
    industry = self.get_industry(stock) or '未知'
    industry_stocks[industry].append((stock, score))

selected = []
for industry, stocks in industry_stocks.items():
    stocks.sort(key=lambda x: x[1], reverse=True)
    selected.append(stocks[0])
```

### 等权调仓

由 `StockSelectionStrategy.rebalance_to()` 自动处理，策略只需返回目标持仓列表。

### 动量排名

```python
def calc_momentum(stock, period=20):
    closes = self.get_close_prices(stock, period + 1)
    if len(closes) >= period + 1 and closes[0] > 0:
        return (closes[-1] - closes[0]) / closes[0]
    return None

ranked = self.rank_stocks(
    lambda s: calc_momentum(s, 20),
    stock_pool=pool,
    top_n=10
)
```

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| 文件路径不存在 | 报错并提示用户检查路径 |
| URL 无法访问 | 报错并提示用户检查链接，建议改用本地文件 |
| PDF/Word 提取库未安装 | 提示用户安装：`pip install pdfplumber` 或 `pip install python-docx` |
| 策略逻辑不清晰或矛盾 | 列出已理解的内容，询问用户确认或补充 |
| 文档中的指标项目不支持 | 用最接近的替代方案，并告知用户差异 |
| 回测运行报错 | 分析错误原因，修复代码后重试，最多重试3次 |
| 策略类型无法判断 | 询问用户选择策略基类 |
| 财务数据字段不确定 | 查阅 `core/financial_data.py` 确认可用字段 |
| 策略注册名与已有策略冲突 | 在名称后加数字后缀或使用更具体的描述 |
| 回测结果为空 | 检查数据源、日期范围、股票池配置 |

## 核心原则

1. **忠实原文**：策略逻辑应忠实于文档描述，不擅自添加文档未提及的逻辑
2. **项目约定**：严格遵循项目的代码风格、注册机制、目录结构
3. **参数可调**：所有关键阈值都应定义为 params，而非硬编码
4. **批量优先**：涉及多只股票的财务数据查询，使用 `get_financial_fields_batch` 而非循环单查
5. **防御性编程**：所有外部数据获取都应处理 None 返回值
6. **日志完善**：关键决策点（选股结果、调仓操作）都应有日志输出
7. **先跑通再优化**：先生成能跑通的基础版本，回测成功后再考虑优化
8. **与 strategy-optimizer 衔接**：策略生成后，可调用 strategy-optimizer 技能进行系统性优化

## 与现有技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `strategy-generator`（本技能） | 从文档生成新策略 | "从0到1" |
| `strategy-optimizer` | 优化已有策略性能 | "从1到优" |

典型工作流：用户用 `strategy-generator` 生成策略 → 回测验证 → 用 `strategy-optimizer` 优化性能。

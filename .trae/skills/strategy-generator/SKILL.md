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
├── generate_report.py                   # HTML报告生成脚本（阶段五创建）
├── readme.md                            # 策略说明文档（阶段四生成，含回测结果）
├── backtest_report.html                 # HTML回测报告（阶段五生成，可视化展示）
└── backtest_results/                    # 回测结果目录（回测时自动生成）
    └── <timestamp>_<strategy_name>.json # 回测记录JSON（含meta/config/metrics/trade_log/equity_curve）
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

使用 `main.py` 命令行方式执行回测，**不要创建独立的 `run_backtest.py` 脚本**。

**命令格式**：

```powershell
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
& "$pythonPath" main.py --mode backtest --strategy <strategy_name> --period 1d --pool <股票池> --start <起始日期> --end <结束日期> --ai-mode
```

**参数说明**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode backtest` | 回测模式（固定） | `--mode backtest` |
| `--strategy` | 策略名称（注册名） | `--strategy twenty_eight_rotation` |
| `--period` | 数据周期 | `--period 1d` |
| `--pool` | 股票池板块名称 | `--pool 中证全指` |
| `--start` | 回测起始日期 | `--start 2020-04-28` |
| `--end` | 回测结束日期 | `--end 2026-04-28` |
| `--ai-mode` | AI模式，跳过图形界面（必须加） | `--ai-mode` |
| `--debug` | 调试模式（可选，输出详细日志） | `--debug` |
| `--no-record` | 禁用回测记录（**不要加**，需要记录结果） | - |

**选股策略示例**：

```powershell
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
& "$pythonPath" main.py --mode backtest --strategy undervalued --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode
```

**择时/轮动策略示例**：

```powershell
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
& "$pythonPath" main.py --mode backtest --strategy twenty_eight_rotation --period 1d --pool 中证全指 --start 2020-04-28 --end 2026-04-28 --ai-mode
```

**关键规则**：
- **必须加 `--ai-mode`**，跳过图形界面渲染
- **不要加 `--no-record`**，让回测结果自动保存到策略目录下的 `backtest_results/` 文件夹
- 回测完成后，`backtest_results/` 目录下会自动生成 `<timestamp>_<strategy_name>.json`，包含完整的 `meta`、`config`、`metrics`、`trade_log`、`equity_curve` 数据
- 阶段四和阶段五的报告生成均从 `backtest_results/` 中读取最新的回测记录
- 命令必须在项目根目录 `e:\jupyter notebook\automatic\qmt_backtrader` 下执行

#### 3.3 回测结果报告

回测完成后，向用户报告以下核心指标：

| 指标 | 说明 |
|------|------|
| 夏普比率 | 风险调整后收益 |
| 总收益率 | 策略总回报 |
| 年化收益率 | 年化后的收益率 |
| 最大回撤 | 最大峰谷回撤 |
| 索提诺比率 | 下行风险调整后收益 |
| 最终权益 | 回测结束时的账户权益 |

### 阶段四：自动生成 readme.md

回测完成后，**必须**自动生成策略目录下的 `readme.md` 文件，包含策略说明和回测结果。

#### 4.1 生成时机

- 回测执行完成后，从 `backtest_results/` 目录读取最新的回测记录 JSON
- 使用最新 JSON 中的 `metrics` 和 `config` 数据填充回测结果部分
- 如果 `backtest_results/` 目录不存在或为空，仍生成 readme.md 但标注"回测未完成"

#### 4.2 读取最新回测记录

```python
import os
import json
import glob

def get_latest_backtest_result(strategy_dir):
    """从 backtest_results/ 目录读取最新的回测结果JSON"""
    results_dir = os.path.join(strategy_dir, 'backtest_results')
    if not os.path.isdir(results_dir):
        return None
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    if not json_files:
        return None
    latest_file = json_files[-1]  # 按文件名排序，最新的在最后
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)
```

#### 4.3 readme.md 模板（选股调仓型）

```markdown
# <策略中文名>

## 策略概述

<策略的核心思想和目标>

## 选股逻辑

1. <步骤1>
2. <步骤2>
3. <步骤3>

## 调仓规则

- <调仓频率>调仓，等权重持仓
- 最多持仓 <N> 只股票
- 仓位比例 <ratio>%

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rebalance_freq | <freq> | 调仓频率 |
| max_stocks | <N> | 最大持仓数量 |
| position_ratio | <ratio> | 仓位比例 |

## 回测结果（<股票池>，<起始日期> ~ <结束日期>）

| 指标 | 数值 |
|------|------|
| 初始资金 | <initial_capital> |
| 最终权益 | <final_value> |
| 总收益率 | <total_return_pct>% |
| 年化收益率 | <annual_return_pct>% |
| 夏普比率 | <sharpe_ratio> |
| 索提诺比率 | <sortino_ratio> |
| 最大回撤 | <max_drawdown_pct>% |
| 交易天数 | <total_trading_days> |
| 盈利天数 / 亏损天数 | <win_days> / <loss_days> |
| 胜率（日） | <win_rate_pct>% |
| 总手续费 | <fee> |
| 换手额 | <turnover> |

## 回测配置

- 初始资金: <initial_capital>
- 股票池: <pool>
- 回测区间: <start_date> ~ <end_date>
- 基准指数: <benchmark>
- 佣金: <commission>

## 快速开始

```bash
python main.py --mode backtest --strategy <strategy_name> --period 1d --pool <股票池> --start <起始日期> --end <结束日期> --ai-mode --no-record
```

## 数据来源

- <数据来源说明>
```

#### 4.4 readme.md 模板（择时/轮动型）

```markdown
# <策略中文名>

## 策略概述

<策略的核心思想和目标>

## 交易逻辑

1. <条件1>时买入
2. <条件2>时卖出

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| <param1> | <default1> | <说明1> |
| <param2> | <default2> | <说明2> |

## 回测结果（<起始日期> ~ <结束日期>）

| 指标 | 数值 |
|------|------|
| 初始资金 | <initial_capital> |
| 最终权益 | <final_value> |
| 总收益率 | <total_return_pct>% |
| 年化收益率 | <annual_return_pct>% |
| 夏普比率 | <sharpe_ratio> |
| 索提诺比率 | <sortino_ratio> |
| 最大回撤 | <max_drawdown_pct>% |
| 交易天数 | <total_trading_days> |
| 盈利天数 / 亏损天数 | <win_days> / <loss_days> |
| 胜率（日） | <win_rate_pct>% |
| 总手续费 | <fee> |
| 换手额 | <turnover> |

## 回测配置

- 初始资金: <initial_capital>
- 回测区间: <start_date> ~ <end_date>
- 基准指数: <benchmark>
- 佣金: <commission>

## 快速开始

```bash
python main.py --mode backtest --strategy <strategy_name> --period 1d --start <起始日期> --end <结束日期> --ai-mode --no-record
```

## 数据来源

- <数据来源说明>
```

#### 4.5 生成方式

读取 `backtest_results/` 目录下最新的回测记录 JSON，从中提取 `metrics` 和 `config` 字段替换模板中的占位符，写入策略目录下的 `readme.md`。

**回测记录 JSON 结构参考**：

```json
{
  "meta": {
    "run_id": "20260516_175424_undervalued",
    "strategy_name": "undervalued",
    "timestamp": "2026-05-16T17:54:24",
    "framework_version": "1.0"
  },
  "config": {
    "cash": 1000000,
    "commission": 0.0013,
    "start_date": "2020-04-28",
    "end_date": "2026-04-28",
    "period": "1d",
    "pool": "中证1000",
    "benchmark": "000852.SH"
  },
  "metrics": {
    "initial_capital": 1000000,
    "final_value": 1866891.37,
    "total_profit": 866891.37,
    "total_return_pct": 86.69,
    "fee": 37099.86,
    "sharpe_ratio": 0.677,
    "max_drawdown_pct": -29.63,
    "annual_return_pct": 11.43,
    "total_trading_days": 1454,
    "win_days": 770,
    "loss_days": 683,
    "turnover": 28538353.83,
    "total_volume": 3302500
  },
  "trade_log": [...],
  "equity_curve": [...]
}
```

**字段映射**：

| readme.md 占位符 | JSON 路径 | 说明 |
|-----------------|----------|------|
| `<initial_capital>` | `metrics.initial_capital` | 初始资金 |
| `<final_value>` | `metrics.final_value` | 最终权益 |
| `<total_return_pct>` | `metrics.total_return_pct` | 总收益率(%) |
| `<annual_return_pct>` | `metrics.annual_return_pct` | 年化收益率(%) |
| `<sharpe_ratio>` | `metrics.sharpe_ratio` | 夏普比率 |
| `<max_drawdown_pct>` | `metrics.max_drawdown_pct` | 最大回撤(%) |
| `<total_trading_days>` | `metrics.total_trading_days` | 交易天数 |
| `<win_days>` | `metrics.win_days` | 盈利天数 |
| `<loss_days>` | `metrics.loss_days` | 亏损天数 |
| `<fee>` | `metrics.fee` | 总手续费 |
| `<turnover>` | `metrics.turnover` | 换手额 |
| `<pool>` | `config.pool` | 股票池 |
| `<start_date>` | `config.start_date` | 回测起始日期 |
| `<end_date>` | `config.end_date` | 回测结束日期 |
| `<benchmark>` | `config.benchmark` | 基准指数 |
| `<commission>` | `config.commission` | 佣金 |

**关键规则**：
- 所有 `<placeholder>` 必须用实际数据替换，不得留空
- 回测结果表格中的数值直接从最新回测记录 JSON 的 `metrics` 和 `config` 读取
- 如果 `backtest_results/` 目录不存在或为空，标注"回测未完成"
- readme.md 写入策略目录根级，与策略 `.py` 文件同级

### 阶段五：自动生成 HTML 回测报告

回测完成后，**必须**自动生成策略目录下的 `backtest_report.html` 文件，提供可视化的策略回测结果展示。

#### 5.1 生成时机

- 紧接 readme.md 生成之后执行
- 从 `backtest_results/` 目录读取最新的回测记录 JSON
- 如果 `backtest_results/` 目录不存在或为空，跳过此阶段

#### 5.2 HTML 报告内容

报告包含以下模块：

| 模块 | 内容 | 数据来源 |
|------|------|---------|
| 策略概览 | 策略名称、回测区间、股票池 | `meta` + `config` 字段 |
| 核心指标卡片 | 总收益率、年化收益率、夏普比率、最大回撤、胜率等 | `metrics` 字段 |
| 权益曲线图 | 策略净值随时间变化 | `equity_curve` 数组 |
| 月度收益热力图 | 按年月展示月度收益率 | 从 `equity_curve` 计算 |
| 交易记录表 | 最近50条交易明细 | `trade_log` 数组 |
| 策略参数表 | 策略使用的参数 | `config` + `strategy_params` 字段 |

#### 5.3 HTML 报告生成脚本

在策略目录下创建 `generate_report.py`，内容如下：

```python
import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)


def get_latest_backtest_result(strategy_dir):
    """从 backtest_results/ 目录读取最新的回测结果JSON"""
    results_dir = os.path.join(strategy_dir, 'backtest_results')
    if not os.path.isdir(results_dir):
        return None
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    if not json_files:
        return None
    latest_file = json_files[-1]
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _heatmap_color(value):
    """根据月度收益率返回背景色"""
    if value is None:
        return 'rgba(148, 163, 184, 0.05)'
    if value > 5:
        return 'rgba(52, 211, 153, 0.7)'
    if value > 2:
        return 'rgba(52, 211, 153, 0.4)'
    if value > 0:
        return 'rgba(52, 211, 153, 0.2)'
    if value > -2:
        return 'rgba(248, 113, 113, 0.2)'
    if value > -5:
        return 'rgba(248, 113, 113, 0.4)'
    return 'rgba(248, 113, 113, 0.7)'


def _heatmap_text_color(value):
    """根据月度收益率返回文字色"""
    if value is None:
        return 'transparent'
    if abs(value) > 2:
        return '#fff'
    return '#94a3b8'


def generate_report():
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    data = get_latest_backtest_result(strategy_dir)

    if not data:
        print(f"未找到回测结果，跳过报告生成")
        return

    # 从回测记录JSON中提取数据
    meta = data.get('meta', {})
    config = data.get('config', {})
    metrics = data.get('metrics', {})
    trade_log = data.get('trade_log', [])
    equity_curve = data.get('equity_curve', [])
    strategy_params = data.get('strategy_params', {})

    strategy_name = meta.get('strategy_name', 'unknown')
    strategy_cn_name = '<策略中文名>'
    start_date = config.get('start_date', '')
    end_date = config.get('end_date', '')
    pool = config.get('pool', '')
    benchmark = config.get('benchmark', '000300.SH')
    commission = config.get('commission', 0)

    initial_capital = metrics.get('initial_capital', 0)
    final_value = metrics.get('final_value', 0)
    total_return_pct = metrics.get('total_return_pct', 0)
    annual_return_pct = metrics.get('annual_return_pct', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    max_drawdown_pct = metrics.get('max_drawdown_pct', 0)
    total_trading_days = metrics.get('total_trading_days', 0)
    win_days = metrics.get('win_days', 0)
    loss_days = metrics.get('loss_days', 0)
    win_rate_pct = round(win_days / (win_days + loss_days) * 100, 1) if (win_days + loss_days) > 0 else 0
    turnover = metrics.get('turnover', 0)
    fee = metrics.get('fee', 0)
    total_profit = metrics.get('total_profit', 0)

    # 计算月度收益率（从 equity_curve）
    monthly_returns = {}
    if equity_curve:
        monthly_values = {}
        for item in equity_curve:
            ym = item['date'][:7]
            monthly_values[ym] = item['portfolio_value']
        sorted_months = sorted(monthly_values.keys())
        for i in range(1, len(sorted_months)):
            prev_val = monthly_values[sorted_months[i - 1]]
            curr_val = monthly_values[sorted_months[i]]
            if prev_val > 0:
                ret = (curr_val - prev_val) / prev_val * 100
                monthly_returns[sorted_months[i]] = round(ret, 2)

    # 月度热力图数据
    heatmap_years = sorted(set(ym[:4] for ym in monthly_returns.keys()))
    heatmap_months_labels = [f'{i}月' for i in range(1, 13)]
    heatmap_data = []
    for year in heatmap_years:
        row = []
        for month in range(1, 13):
            ym = f'{year}-{month:02d}'
            row.append(monthly_returns.get(ym, None))
        heatmap_data.append(row)

    # 交易记录表格（最近50条）
    recent_trades = trade_log[-50:] if len(trade_log) > 50 else trade_log
    trade_rows = ""
    for t in recent_trades:
        direction = '买入' if t.get('direction') == '0' else '卖出'
        dir_class = 'trade-buy' if direction == '买入' else 'trade-sell'
        pnl_val = t.get('pnl', 0)
        pnl_class = 'up' if pnl_val > 0 else ('down' if pnl_val < 0 else '')
        pnl_str = f'{pnl_val:+.2f}' if pnl_val != 0 else '-'
        trade_time = t.get('trade_time', '')[:10]
        trade_rows += f'''<tr>
            <td>{trade_time}</td>
            <td>{t.get('instrument_id', '')}</td>
            <td class="{dir_class}">{direction}</td>
            <td>{t.get('trade_price', 0):.2f}</td>
            <td>{t.get('volume', 0)}</td>
            <td class="{pnl_class}">{pnl_str}</td>
            <td>{t.get('memo', '')}</td>
        </tr>'''

    # 权益曲线数据
    equity_dates = json.dumps([d['date'] for d in equity_curve], ensure_ascii=False)
    equity_values = json.dumps([d['portfolio_value'] for d in equity_curve], ensure_ascii=False)

    # 热力图HTML片段（预先计算）
    heatmap_header_html = " ".join(f'<div class="heatmap-header">{m}</div>' for m in heatmap_months_labels)
    heatmap_body_html = ""
    for y, row in zip(heatmap_years, heatmap_data):
        heatmap_body_html += f'<div class="heatmap-label">{y}</div>'
        for v in row:
            bg = _heatmap_color(v)
            tc = _heatmap_text_color(v)
            text = f'{v:.1f}' if v is not None else ''
            heatmap_body_html += f'<div class="heatmap-cell" style="background:{bg};color:{tc}">{text}</div>'

    # 策略参数展示
    params_html = ""
    for k, v in strategy_params.items():
        if k in ('stock_pool', 'instrument_id', 'exchange', 'kline_style'):
            continue
        params_html += f'<div class="param-item"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>'

    # 收益率颜色判断
    return_color = '#34d399' if total_return_pct > 0 else '#f87171'
    sharpe_color = '#34d399' if sharpe_ratio > 0 else '#f87171'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_cn_name} - 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 2.2em; text-align: center; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 1.4em; color: #60a5fa; margin: 40px 0 20px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }}
.subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 40px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 30px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.8em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 1.8em; font-weight: 700; margin: 6px 0; }}
.up {{ color: #34d399; }}
.down {{ color: #f87171; }}
.chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin: 24px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: #1e293b; border-radius: 12px; overflow: hidden; }}
th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; padding: 12px 14px; text-align: left; }}
td {{ padding: 10px 14px; border-top: 1px solid #334155; font-size: 0.9em; }}
.trade-buy {{ color: #34d399; font-weight: 600; }}
.trade-sell {{ color: #f87171; font-weight: 600; }}
.heatmap-grid {{ display: grid; grid-template-columns: 60px repeat(12, 1fr); gap: 3px; margin: 16px 0; }}
.heatmap-label {{ font-size: 0.75em; color: #94a3b8; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; }}
.heatmap-header {{ font-size: 0.7em; color: #94a3b8; text-align: center; padding: 4px; }}
.heatmap-cell {{ border-radius: 4px; text-align: center; font-size: 0.7em; font-weight: 600; padding: 6px 2px; min-height: 28px; display: flex; align-items: center; justify-content: center; }}
.params-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }}
.param-item {{ background: #0f172a; border-radius: 8px; padding: 10px 14px; display: flex; justify-content: space-between; }}
.param-key {{ color: #94a3b8; font-size: 0.85em; }}
.param-val {{ color: #60a5fa; font-weight: 600; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

<h1>{strategy_cn_name}</h1>
<p class="subtitle">回测区间: {start_date} 至 {end_date} | 股票池: {pool or "默认"} | 基准: {benchmark}</p>

<h2>一、核心指标</h2>
<div class="summary-cards">
  <div class="card">
    <div class="card-label">总收益率</div>
    <div class="card-value" style="color:{return_color}">{total_return_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">年化收益率</div>
    <div class="card-value" style="color:{return_color}">{annual_return_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">夏普比率</div>
    <div class="card-value" style="color:{sharpe_color}">{sharpe_ratio:.4f}</div>
  </div>
  <div class="card">
    <div class="card-label">最大回撤</div>
    <div class="card-value down">{max_drawdown_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">日胜率</div>
    <div class="card-value">{win_rate_pct:.1f}%</div>
  </div>
</div>

<div class="summary-cards">
  <div class="card">
    <div class="card-label">初始资金</div>
    <div class="card-value" style="font-size:1.2em">{initial_capital:,.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">最终权益</div>
    <div class="card-value" style="font-size:1.2em;color:{return_color}">{final_value:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">总利润</div>
    <div class="card-value" style="font-size:1.2em;color:{return_color}">{total_profit:+,.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">交易天数</div>
    <div class="card-value" style="font-size:1.2em">{total_trading_days}</div>
  </div>
  <div class="card">
    <div class="card-label">盈利/亏损天数</div>
    <div class="card-value" style="font-size:1.2em"><span class="up">{win_days}</span> / <span class="down">{loss_days}</span></div>
  </div>
  <div class="card">
    <div class="card-label">总手续费</div>
    <div class="card-value" style="font-size:1.2em">{fee:,.2f}</div>
  </div>
</div>

<h2>二、权益曲线</h2>
<div class="chart-box">
  <canvas id="chartEquity"></canvas>
</div>

<h2>三、月度收益热力图</h2>
<div class="chart-box">
  <div class="heatmap-grid">
    <div class="heatmap-header"></div>
    {heatmap_header_html}
    {heatmap_body_html}
  </div>
</div>

<h2>四、交易记录（最近50条）</h2>
<table>
  <thead>
    <tr><th>日期</th><th>标的</th><th>方向</th><th>价格</th><th>数量</th><th>盈亏</th><th>备注</th></tr>
  </thead>
  <tbody>{trade_rows or '<tr><td colspan="7" style="text-align:center;color:#94a3b8">暂无交易记录</td></tr>'}</tbody>
</table>

<h2>五、策略参数</h2>
<div class="params-grid">
  {params_html or '<div style="color:#94a3b8">无自定义参数</div>'}
</div>

</div>

<script>
Chart.defaults.color = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.1)';

new Chart(document.getElementById('chartEquity'), {{
  type: 'line',
  data: {{
    labels: {equity_dates},
    datasets: [{{
      label: '账户权益',
      data: {equity_values},
      borderColor: '#60a5fa',
      backgroundColor: 'rgba(96, 165, 250, 0.1)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0,
      tension: 0.1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: '策略权益曲线' }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{ return '权益: ' + ctx.parsed.y.toLocaleString(); }}
        }}
      }}
    }},
    scales: {{
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{
          callback: function(v) {{ return (v/10000).toFixed(0) + '万'; }}
        }}
      }},
      x: {{
        grid: {{ display: false }},
        ticks: {{ maxTicksLimit: 12 }}
      }}
    }}
  }}
}});
</script>
</body>
</html>'''

    output_file = os.path.join(strategy_dir, 'backtest_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML报告已生成: {output_file}')


if __name__ == '__main__':
    generate_report()
```

#### 5.4 HTML 报告生成执行

回测完成后（`main.py --mode backtest` 执行结束），使用以下命令生成报告：

```powershell
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
& "$pythonPath" "<策略目录>/generate_report.py"
```

**注意**：
- `generate_report.py` 会自动从 `backtest_results/` 目录读取最新的回测记录 JSON，无需手动指定文件路径
- 必须在回测完成后再执行此脚本，否则 `backtest_results/` 目录中没有数据

#### 5.5 HTML 报告设计规范

| 规范 | 说明 |
|------|------|
| 配色 | 深色主题（#0f172a 背景），与项目现有优化报告风格一致 |
| 图表库 | Chart.js 4.x（CDN加载） |
| 响应式 | 支持不同屏幕宽度，卡片网格自适应 |
| 数据来源 | 全部从 `backtest_results/` 目录最新 JSON 读取，不硬编码 |
| 文件大小 | 控制在 500KB 以内（权益曲线数据采样如果超过1000点则降采样） |

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

典型工作流：用户用 `strategy-generator` 生成策略 → 回测验证 → 自动生成 readme.md + HTML报告 → 用 `strategy-optimizer` 优化性能。

# 使用方法

## 1. 运行回测

**小市值策略回测：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug
```

**特质波动率因子策略回测：**
```bash
python main.py --mode backtest --strategy ivff3 --period 1d --pool 中证1000 --start 2015-01-01 --end 2026-04-28 --debug
```

**低估价值策略回测：**
```bash
python main.py --mode backtest --strategy undervalued --period 1d --pool 沪深300 --start 2020-04-28 --end 2026-04-28 --debug
```

**银行轮动策略：**
```bash
python main.py --mode backtest --strategy bank_rotation --period 1d --start 2020-04-28 --end 2026-04-28 --ai-mode
```

**指定数据源回测：**
```bash
# OpenData 数据源（免费，无需 QMT）
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --data-source open

# 富途数据源（需安装 futu-api 并启动 OpenD）
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --data-source futu
```

**AI 模式回测（跳过图形界面，适用于自动化优化）：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record
```

**使用不复权数据回测（适用于打板、涨停检测等策略）：**
```bash
# 策略在 @register_strategy 的 default_kwargs 中指定 dividend_type='none'，
# 运行方式与普通回测相同：
python main.py --mode backtest --strategy <策略名> --ai-mode
```

> 不复权数据回测需要预先下载不复权数据到 `market_raw` 目录：
> ```bash
> python download_qmt_market_data.py --pool 中证全指 --type market_raw --period 1d
> ```
> 详见 [策略开发文档](strategy-development.md#使用不复权数据回测)。

**使用 YAML 配置文件：**
```bash
python main.py --mode backtest --strategy small_cap --config config/backtest.yaml
```

**参数说明**：
- `--mode`：运行模式，可选值：`backtest`（回测）、`sim`（模拟交易）、`real`（实盘交易）、`instances`（多策略实例）
- `--strategy`：策略类型，可选值：`small_cap`、`ivff3`、`undervalued`、`jq_small_cap`、`bank_rotation`、`dividend_value_growth`、`medical_multi_factor`、`twenty_eight_rotation`、`first_board_low_open`、`rothman_value` 等（详见 [策略开发文档](strategy-development.md#内置策略)）
- `--period`：数据周期，可选值：`1d`（日线）、`1m`、`5m`、`15m`、`30m`、`60m`、`tick`
- `--pool`：股票池板块名称，如 `沪深300`、`沪深A股`、`上证50`、`中证500`、`中证1000`、`中小综指`
- `--start`：回测起始日期，格式：`YYYY-MM-DD`
- `--end`：回测结束日期，格式：`YYYY-MM-DD`
- `--data-source`：行情数据源，可选值：`qmt`（默认）、`open`（OpenData）、`futu`（富途）
- `--config`：YAML 配置文件路径，配置覆盖默认值，命令行参数覆盖 YAML 配置
- `--qmt-path`：QMT userdata_mini 路径（默认 `D:\qmt\userdata_mini`）
- `--account`：QMT 资金账号，不传则自动获取第一个
- `--instances`：策略实例配置文件路径（JSON），用于 `--mode instances` 模式
- `--cache-dir`：自定义缓存数据存储目录（默认项目根目录下 `.cache`）
- `--mem-limit`：内存缓存最大对象数量限制（默认 500）
- `--debug`：启用 DEBUG 日志模式，输出详细调试信息
- `--ai-mode`：启用 AI 自动运行模式，跳过所有图形界面渲染，适用于自动化策略优化
- `--no-record`：禁用回测结果自动记录到本地文件
- `--slippage`：滑点百分比，如 `0.001` 表示 0.1%，不传则使用策略默认值
- `--strategy-params`：策略参数动态覆盖，格式为 JSON 字符串，如 `"symbol=000001.SZ"` 或 `'{"max_stocks": 20}'`，覆盖策略 `default_kwargs` 中的同名参数

## 2. 查看回测结果

回测结束后，框架会根据运行模式自动展示结果：

### GUI 模式（默认）

不带 `--ai-mode` 参数运行回测时，回测结束后会自动弹出基于 PyQt5 + pyqtgraph 的回测报告窗口。

**启动方式：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28
```

**窗口包含 7 个标签页：**

| 标签页 | 内容 |
|--------|------|
| **总览** | 核心指标面板（整体表现、风险评估、交易统计、基准分析）+ 权益曲线图 |
| **K线图** | 日K线蜡烛图 + 均线（MA5/MA10/MA20/MA60）+ 买卖标记 + 成交量 + 十字光标悬浮提示 |
| **每日收益** | 每日盈亏柱状图（红色盈利、绿色亏损）+ 十字光标悬浮提示 |
| **交易明细** | 交易统计面板 + 交易记录表格（支持排序、列筛选、标的显隐切换） |
| **每日持仓收益** | 持仓收益统计面板 + 每日持仓明细表格（支持排序、列筛选） |
| **基准对比** | 策略与基准的归一化收益曲线对比图 |
| **换手率评估** | 换手率统计面板（日/周/月）+ 换手率趋势图 |

**交互功能：**

- **K线图**：鼠标滚轮缩放、拖拽平移、向左拖拽加载更多历史数据、十字光标显示 OHLC 信息
- **权益曲线/基准对比**：鼠标悬浮显示日期和数值、滚轮缩放、拖拽平移
- **交易明细表格**：点击列头排序、列头右侧筛选按钮按值过滤、`隐藏标的` 按钮切换标的列显隐
- **每日持仓收益表格**：同交易明细，支持排序、筛选、标的显隐

> **注意**：K线图标签页仅在策略提供了 `instrument_id`、`exchange`、`kline_style` 参数时显示；多标的选股策略会自动使用基准数据生成K线图。

### AI 模式

添加 `--ai-mode` 参数时，跳过所有 GUI 渲染，仅在终端输出回测结果摘要：

```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode
```

适用于自动化策略优化、CI/CD 流水线、远程服务器等无图形界面的场景。

### HTML 可视化报告

框架支持基于 Plotly 的 HTML 可视化报告，可在浏览器中查看。通过回测结果记录器自动生成，详见 [回测结果记录文档](backtest-recorder.md)。

## 3. 运行模拟交易

```bash
python main.py --mode sim --strategy small_cap --qmt-path D:\qmt\userdata_mini
```

## 4. 运行实盘交易

```bash
python main.py --mode real --strategy small_cap --qmt-path D:\qmt\userdata_mini --account 12345678
```

## 5. 运行多策略实例

当需要在同一账户下同时运行多个策略时，使用多策略实例模式。每个策略实例拥有独立的虚拟持仓簿，实现持仓和资金的策略级隔离。

**创建配置文件** `config/instances.json`：

```json
{
  "instances": [
    {
      "instance_id": "small_cap_sim",
      "strategy_name": "small_cap",
      "mode": "sim",
      "account_id": "12345678",
      "initial_capital": 500000,
      "claim_existing_positions": true,
      "kwargs": {}
    },
    {
      "instance_id": "undervalued_sim",
      "strategy_name": "undervalued",
      "mode": "sim",
      "account_id": "12345678",
      "initial_capital": 500000,
      "claim_existing_positions": true,
      "kwargs": {}
    }
  ]
}
```

**配置说明**：
- `instance_id`：策略实例唯一标识，用于日志隔离和订单路由
- `strategy_name`：策略名称，对应 `--strategy` 参数的可选值
- `mode`：运行模式，`sim`（模拟）或 `real`（实盘）
- `account_id`：QMT 资金账号，相同账户的策略共享一个 QMTAPI 实例
- `initial_capital`：策略初始资金，用于虚拟簿记
- `claim_existing_positions`：是否认领账户现有持仓（首次启动时按标的分配给策略）
- `kwargs`：策略参数，覆盖默认参数

**启动多策略**：

```bash
python main.py --mode instances --instances config/instances.json
```

> **注意**：单策略模式（`sim`/`real`）也默认启用虚拟簿记，无需额外配置即可享受策略级隔离。

## 6. 数据预下载

在运行回测前，可以预先下载行情和财务数据到本地缓存，避免回测时逐只下载导致等待过长：

**下载行情数据（OpenData 数据源）：**
```bash
python download_open_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-04-28
```

**下载行情数据（QMT 数据源）：**
```bash
python download_qmt_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-04-28
```

**下载行情数据（富途数据源）：**
```bash
python download_futu_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-04-28
```

**下载财务数据：**
```bash
python download_qmt_financial_data.py --pool 中证1000 --start 2020-01-01
```

**清理过期日志：**
```bash
clean_old_logs.bat
```

## 7. Web 回测查看器

框架提供了基于 Flask + Vue3 + ECharts 的 Web 界面，用于浏览和对比策略回测结果。

**启动方式：**

方式一：双击批处理文件
```bash
start_web.bat
```

方式二：命令行启动
```bash
cd web
python app.py
```

启动后浏览器访问 **http://localhost:5000** 即可使用。

**功能说明：**

| 功能 | 说明 |
|------|------|
| 策略列表 | 自动发现 `strategies/` 和 `strategies_for_vip/` 下含回测结果的策略，按标准/VIP 分组显示 |
| 回测记录 | 按时间倒序列出同一策略的所有回测记录，显示收益率和夏普比率 |
| 详情查看 | 查看单次回测的核心指标（总收益率、年化收益率、夏普比率、最大回撤、胜率等）、权益曲线、回撤曲线、交易日志 |
| 多次对比 | 对比同一策略多次回测的夏普比率、收益率、最大回撤柱状图 |
| 策略逻辑 | 查看策略的 README 文档（Markdown 渲染） |
| 策略代码 | 查看回测时保存的策略源代码（仅新版回测结果包含） |
| 自定义名称 | 双击策略名或点击编辑按钮可修改策略显示名称 |
| 删除记录 | 鼠标悬停回测记录可删除指定回测结果 |

# 使用方法

## 1. 运行回测

**高股息策略回测：**
```bash
python main.py --mode backtest --strategy high_dividend --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug
```

**小市值策略回测：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --debug
```

**AI 模式回测（跳过图形界面，适用于自动化优化）：**
```bash
python main.py --mode backtest --strategy small_cap --period 1d --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --ai-mode --no-record
```

**参数说明**：
- `--mode`：运行模式，可选值：`backtest`（回测）、`sim`（模拟交易）、`real`（实盘交易）、`instances`（多策略实例）
- `--strategy`：策略类型，可选值：`high_dividend`、`small_cap` 等
- `--period`：数据周期，可选值：`1d`（日线）、`1m`、`5m`、`15m`、`30m`、`60m`、`tick`
- `--pool`：股票池板块名称，如 `沪深300`、`沪深A股`、`上证50`、`中证500`、`中证1000`
- `--start`：回测起始日期，格式：`YYYY-MM-DD`
- `--end`：回测结束日期，格式：`YYYY-MM-DD`
- `--qmt-path`：QMT userdata_mini 路径（默认 `D:\qmt\userdata_mini`）
- `--account`：QMT 资金账号，不传则自动获取第一个
- `--instances`：策略实例配置文件路径（JSON），用于 `--mode instances` 模式
- `--cache-dir`：自定义缓存数据存储目录（默认项目根目录下 `.cache`）
- `--mem-limit`：内存缓存最大对象数量限制（默认 500）
- `--debug`：启用 DEBUG 日志模式，输出详细调试信息
- `--ai-mode`：启用 AI 自动运行模式，跳过所有图形界面渲染，适用于自动化策略优化
- `--no-record`：禁用回测结果自动记录到本地文件
- `--slippage`：滑点百分比，如 `0.001` 表示 0.1%，不传则使用策略默认值

## 2. 运行模拟交易

```bash
python main.py --mode sim --strategy high_dividend --qmt-path D:\qmt\userdata_mini
```

## 3. 运行实盘交易

```bash
python main.py --mode real --strategy high_dividend --qmt-path D:\qmt\userdata_mini --account 12345678
```

## 4. 运行多策略实例

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
      "instance_id": "high_div_sim",
      "strategy_name": "high_dividend",
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

## 5. 数据预下载

在运行回测前，可以预先下载行情和财务数据到本地缓存，避免回测时逐只下载导致等待过长：

**下载行情数据：**
```bash
python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-04-28
```

**下载财务数据：**
```bash
python download_financial_data.py --pool 中证1000 --start 2020-01-01
```

**查看缓存文件内容：**
```bash
python read_parquet.py
```

**清理过期日志：**
```bash
clean_old_logs.bat
```

## 6. Web 回测查看器

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

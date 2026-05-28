# 银行轮动策略 (Bank Rotation Strategy)

## 策略逻辑

复现聚宽囚徒的银行轮动策略，原策略年化收益约77%（2015-2016年回测），无止损机制。

1. 标的：四大银行股
   - 工商银行 (601398.SH)
   - 农业银行 (601288.SH)
   - 建设银行 (601939.SH)
   - 中国银行 (601988.SH)
2. 比率计算：当前价格 / 上一交易日收盘价
   - **日线模式**：当前价格 = 当日收盘价，每日检查一次
   - **分钟线模式**：当前价格 = 当前分钟收盘价，每分钟检查一次
3. 交易规则：
   - **空仓时**：当4只银行股的最大比率与最小比率之差 > 阈值(0.005)，买入比率最小的银行股（全仓）
   - **持仓时**：当持仓股比率与最小比率之差 > 阈值(0.005)，卖出持仓，买入比率最小的银行股
   - **T+1检查**：持仓换仓前检查可卖数量，当天买入的股票不可卖出
4. 核心思想：买入当日相对跌幅最大的银行股，利用四大银行股之间的均值回归效应
5. 交易细节：
   - 买入时预留0.1%资金余量防止手续费不足
   - 遵守A股T+1规则，当天买入的股票不可卖出

<br />

## 回测结果 (2020-04-28 \~ 2026-04-28)

| 指标 | 分钟线模式 (1m) |
|------|-----------------|
| 夏普比率 | 2.331 |
| 总收益率 | 327.13% |
| 年化收益率 | 28.64% |
| 最大回撤 | -15.08% |
| 最终权益 | 854,258.02 |
| 总交易天数 | 1,453 |
| 盈利天数 | 797 |
| 亏损天数 | 570 |
| 总手续费 | 135,399.02 |
| 总成交量 | 79,113,600 |

> 注：回测参数：初始资金200,000，手续费率0.02%，对标基准000300.SH

## 参数说明

| 参数                | 默认值   | 说明              |
| ----------------- | ----- | --------------- |
| spread\_threshold | 0.005 | 比率差阈值，低于此值不触发交易 |

## 默认回测配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 初始资金 | 200,000 | 回测起始资金 |
| 手续费率 | 0.02% | 单边手续费 |
| 起始日期 | 2020-04-28 | 回测开始日期 |
| 结束日期 | 2026-04-28 | 回测结束日期 |
| K线周期 | 1d (日线) | 支持 1d(日线) / 1m(分钟线) |
| 对标基准 | 000300.SH | 沪深300指数 |
| 对比标的 | 四大银行股 | 报告中对比较用的标的列表 |

## 使用方式

### 命令行回测

```bash
# 日线模式回测
python main.py --mode backtest --strategy bank_rotation --period 1d --start 2020-04-28 --end 2026-04-28 --data-source futu --debug

# 分钟线模式回测（需MiniQMT环境）
python main.py --mode backtest --strategy bank_rotation --period 1m --start 2020-04-28 --end 2026-04-28 --data-source futu --debug
```

### 代码调用

```python
from api.backtest_api import BacktestAPI
from strategies import get_strategy, get_strategy_default_kwargs, get_strategy_backtest_config

sc = get_strategy('bank_rotation')
dk = get_strategy_default_kwargs('bank_rotation')
bc = get_strategy_backtest_config('bank_rotation')

api = BacktestAPI(data_source='futu')
api.set_ai_mode(True)
api.configure(**bc)
api.add_strategy(sc, **dk)
results = api.run()
```

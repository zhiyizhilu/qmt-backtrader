import sys
import os
import re
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.data.qmt import QMTDataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_DATE = '2020-04-28'
END_DATE = '2026-04-28'
BUY_TIME = '09:31'
MORNING_SELL_TIME = '11:28'
AFTERNOON_SELL_TIME = '14:50'


def load_screened_stocks(strategy_dir=None):
    if strategy_dir is None:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(strategy_dir, 'screened_stocks_raw.txt')
    stocks = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^(\S+):\s+\d+\s+次,\s+日期:\s+(.*)', line)
            if not m:
                continue
            symbol = m.group(1)
            dates = [d.strip() for d in m.group(2).split(',') if d.strip()]
            stocks[symbol] = dates
    return stocks


def simulate_trade(dp, symbol, buy_date_str):
    try:
        buy_dt = pd.Timestamp(buy_date_str)
        start_dt = buy_dt - pd.Timedelta(days=5)
        sell_dt = buy_dt + pd.Timedelta(days=10)
        df_raw = dp.get_raw_data(symbol, start_dt.strftime('%Y-%m-%d'), sell_dt.strftime('%Y-%m-%d'), period='1m')
        if df_raw is None or df_raw.empty:
            return None

        buy_day_data = df_raw[df_raw.index.date == buy_dt.date()]
        if buy_day_data.empty:
            return None

        buy_bar = None
        for idx, row in buy_day_data.iterrows():
            time_str = idx.strftime('%H:%M')
            if time_str >= BUY_TIME and buy_bar is None:
                buy_bar = row
                buy_time = idx
                break

        if buy_bar is None:
            return None

        buy_price = buy_bar['open']
        if buy_price <= 0:
            return None

        all_dates = sorted(set(df_raw.index.date))
        next_dates = [d for d in all_dates if d > buy_dt.date()]
        if not next_dates:
            return None
        sell_date = next_dates[0]

        sell_day_data = df_raw[df_raw.index.date == sell_date]
        if sell_day_data.empty:
            return None

        morning_sell_bar = None
        morning_sell_time = None
        for idx, row in sell_day_data.iterrows():
            time_str = idx.strftime('%H:%M')
            if time_str >= MORNING_SELL_TIME:
                if morning_sell_bar is None:
                    morning_sell_bar = row
                    morning_sell_time = idx

        afternoon_sell_bar = None
        afternoon_sell_time = None
        for idx, row in sell_day_data.iterrows():
            time_str = idx.strftime('%H:%M')
            if time_str >= AFTERNOON_SELL_TIME:
                if afternoon_sell_bar is None:
                    afternoon_sell_bar = row
                    afternoon_sell_time = idx

        if morning_sell_bar is not None and morning_sell_bar['close'] > buy_price:
            sell_price = morning_sell_bar['close']
            sell_time = morning_sell_time
            sell_reason = 'T+1上午盈利卖出'
        elif afternoon_sell_bar is not None:
            sell_price = afternoon_sell_bar['close']
            sell_time = afternoon_sell_time
            sell_reason = 'T+1尾盘清仓'
        else:
            last_bar = sell_day_data.iloc[-1]
            sell_price = last_bar['close']
            sell_time = sell_day_data.index[-1]
            sell_reason = 'T+1收盘卖出'

        profit_pct = (sell_price - buy_price) / buy_price

        return {
            'symbol': symbol,
            'buy_date': buy_date_str,
            'buy_time': buy_time.strftime('%Y-%m-%d %H:%M'),
            'buy_price': round(buy_price, 2),
            'sell_time': sell_time.strftime('%Y-%m-%d %H:%M'),
            'sell_price': round(sell_price, 2),
            'profit_pct': round(profit_pct * 100, 4),
            'sell_reason': sell_reason,
        }
    except Exception as e:
        logger.debug("%s %s: 模拟交易失败: %s" % (symbol, buy_date_str, e))
        return None


def main():
    strategy_dir = sys.argv[1] if len(sys.argv) > 1 else None

    dp = QMTDataProcessor(use_opendata=True)

    stocks = load_screened_stocks(strategy_dir)
    logger.info("加载 %d 只筛选股票" % len(stocks))

    all_trades = []
    total_signals = 0
    total_simulated = 0
    total_failed = 0

    for i, (symbol, dates) in enumerate(sorted(stocks.items())):
        for buy_date in dates:
            total_signals += 1
            result = simulate_trade(dp, symbol, buy_date)
            if result:
                all_trades.append(result)
                total_simulated += 1
            else:
                total_failed += 1

        if (i + 1) % 20 == 0 or i == len(stocks) - 1:
            logger.info("  进度: %d/%d, 已模拟 %d, 失败 %d" % (
                i + 1, len(stocks), total_simulated, total_failed))

    if not all_trades:
        logger.error("没有成功的交易记录！")
        return

    df = pd.DataFrame(all_trades)

    wins = df[df['profit_pct'] > 0]
    losses = df[df['profit_pct'] <= 0]

    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    avg_win = wins['profit_pct'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['profit_pct'].mean()) if len(losses) > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    total_profit_pct = df['profit_pct'].sum()
    avg_profit_pct = df['profit_pct'].mean()
    max_win = df['profit_pct'].max()
    max_loss = df['profit_pct'].min()

    morning_sells = df[df['sell_reason'] == 'T+1上午盈利卖出']
    afternoon_sells = df[df['sell_reason'] == 'T+1尾盘清仓']

    print("\n" + "=" * 80)
    print("首板低开策略 - 概率统计回测结果")
    print("=" * 80)
    print("回测区间: %s ~ %s" % (START_DATE, END_DATE))
    print("总信号数: %d" % total_signals)
    print("成功模拟: %d" % total_simulated)
    print("模拟失败: %d" % total_failed)
    print()
    print("--- 整体统计 ---")
    print("总交易次数: %d" % len(df))
    print("盈利次数: %d" % len(wins))
    print("亏损次数: %d" % len(losses))
    print("胜率: %.2f%%" % win_rate)
    print("平均盈利: %.4f%%" % avg_win)
    print("平均亏损: %.4f%%" % (-avg_loss if avg_loss > 0 else 0))
    print("盈亏比: %.4f" % profit_loss_ratio)
    print("总收益(单次累加): %.4f%%" % total_profit_pct)
    print("平均单次收益: %.4f%%" % avg_profit_pct)
    print("最大单次盈利: %.4f%%" % max_win)
    print("最大单次亏损: %.4f%%" % max_loss)
    print()
    print("--- 卖出方式统计 ---")
    print("上午盈利卖出: %d 次 (%.1f%%), 平均盈利 %.4f%%" % (
        len(morning_sells), len(morning_sells) / len(df) * 100,
        morning_sells['profit_pct'].mean() if len(morning_sells) > 0 else 0))
    print("尾盘清仓: %d 次 (%.1f%%), 平均收益 %.4f%%" % (
        len(afternoon_sells), len(afternoon_sells) / len(df) * 100,
        afternoon_sells['profit_pct'].mean() if len(afternoon_sells) > 0 else 0))

    if strategy_dir is None:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(strategy_dir, '分钟回测结果.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info("交易明细已保存到: %s" % output_csv)

    print("\n--- 按年份统计 ---")
    df['year'] = pd.to_datetime(df['buy_date']).dt.year
    for year, group in df.groupby('year'):
        y_wins = group[group['profit_pct'] > 0]
        y_losses = group[group['profit_pct'] <= 0]
        y_win_rate = len(y_wins) / len(group) * 100 if len(group) > 0 else 0
        y_avg = group['profit_pct'].mean()
        print("  %d: 交易%d次, 胜率%.1f%%, 平均收益%.4f%%" % (year, len(group), y_win_rate, y_avg))

    print("\n--- 盈利分布 ---")
    bins = [-100, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 100]
    labels = ['<-5%', '-5~-3%', '-3~-2%', '-2~-1%', '-1~0%', '0~1%', '1~2%', '2~3%', '3~5%', '5~10%', '>10%']
    df['profit_bin'] = pd.cut(df['profit_pct'], bins=bins, labels=labels)
    for label in labels:
        count = len(df[df['profit_bin'] == label])
        if count > 0:
            pct = count / len(df) * 100
            print("  %s: %d 次 (%.1f%%)" % (label, count, pct))


if __name__ == '__main__':
    main()

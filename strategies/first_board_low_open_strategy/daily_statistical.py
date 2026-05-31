import sys
import os
import re
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.data.qmt import QMTDataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_DATE = '2020-04-28'
END_DATE = '2026-04-28'


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


def simulate_daily_trade(dp, symbol, buy_date_str):
    try:
        buy_dt = pd.Timestamp(buy_date_str)
        start_dt = buy_dt - pd.Timedelta(days=10)
        end_dt = buy_dt + pd.Timedelta(days=10)
        df = dp.get_raw_data(symbol, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'), period='1d')
        if df is None or df.empty:
            return None

        buy_idx = None
        for i, idx in enumerate(df.index):
            if idx.date() == buy_dt.date():
                buy_idx = i
                break

        if buy_idx is None:
            return None

        buy_price = df.iloc[buy_idx]['open']
        if buy_price <= 0:
            return None

        if buy_idx + 1 >= len(df):
            return None

        next_day = df.iloc[buy_idx + 1]
        next_open = next_day['open']
        next_close = next_day['close']

        if next_open > buy_price:
            sell_price = next_open
            sell_date = df.index[buy_idx + 1].strftime('%Y-%m-%d')
            sell_type = 'T+1开盘卖出(盈利)'
        else:
            sell_price = next_close
            sell_date = df.index[buy_idx + 1].strftime('%Y-%m-%d')
            sell_type = 'T+1收盘卖出'

        profit_pct = (sell_price - buy_price) / buy_price

        return {
            'symbol': symbol,
            'buy_date': buy_date_str,
            'buy_price': round(buy_price, 2),
            'sell_date': sell_date,
            'sell_price': round(sell_price, 2),
            'profit_pct': round(profit_pct * 100, 4),
            'sell_reason': sell_type,
        }
    except Exception as e:
        logger.debug("%s %s: 模拟失败: %s" % (symbol, buy_date_str, e))
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
            result = simulate_daily_trade(dp, symbol, buy_date)
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

    open_sells = df[df['sell_reason'] == 'T+1开盘卖出(盈利)']
    close_sells = df[df['sell_reason'] == 'T+1收盘卖出']

    print("\n" + "=" * 80)
    print("首板低开策略 - 日线统计回测结果")
    print("=" * 80)
    print("回测区间: %s ~ %s" % (START_DATE, END_DATE))
    print("规则: T日开盘买入, T+1开盘盈利则卖出, 否则T+1收盘卖出")
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
    print("T+1开盘卖出(盈利): %d 次 (%.1f%%), 平均盈利 %.4f%%" % (
        len(open_sells), len(open_sells) / len(df) * 100,
        open_sells['profit_pct'].mean() if len(open_sells) > 0 else 0))
    print("T+1收盘卖出: %d 次 (%.1f%%), 平均收益 %.4f%%" % (
        len(close_sells), len(close_sells) / len(df) * 100,
        close_sells['profit_pct'].mean() if len(close_sells) > 0 else 0))

    if strategy_dir is None:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(strategy_dir, '日回测结果.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info("交易明细已保存到: %s" % output_csv)

    print("\n--- 按年份统计 ---")
    df['year'] = pd.to_datetime(df['buy_date']).dt.year
    for year, group in df.groupby('year'):
        y_wins = group[group['profit_pct'] > 0]
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

    print("\n--- 与分钟线统计对比 ---")
    min_csv = os.path.join(strategy_dir, '分钟回测结果.csv')
    if os.path.exists(min_csv):
        df_min = pd.read_csv(min_csv, encoding='utf-8-sig')
        min_wins = df_min[df_min['profit_pct'] > 0]
        min_losses = df_min[df_min['profit_pct'] <= 0]
        min_win_rate = len(min_wins) / len(df_min) * 100 if len(df_min) > 0 else 0
        min_avg_win = min_wins['profit_pct'].mean() if len(min_wins) > 0 else 0
        min_avg_loss = abs(min_losses['profit_pct'].mean()) if len(min_losses) > 0 else 0
        min_plr = min_avg_win / min_avg_loss if min_avg_loss > 0 else float('inf')
        print("指标              日线统计        分钟线统计")
        print("交易次数          %d              %d" % (len(df), len(df_min)))
        print("胜率              %.2f%%          %.2f%%" % (win_rate, min_win_rate))
        print("平均盈利          %.4f%%        %.4f%%" % (avg_win, min_avg_win))
        print("平均亏损          %.4f%%        %.4f%%" % (-avg_loss if avg_loss > 0 else 0, -min_avg_loss if min_avg_loss > 0 else 0))
        print("盈亏比            %.4f          %.4f" % (profit_loss_ratio, min_plr))
        print("平均单次收益      %.4f%%        %.4f%%" % (avg_profit_pct, df_min['profit_pct'].mean()))


if __name__ == '__main__':
    main()

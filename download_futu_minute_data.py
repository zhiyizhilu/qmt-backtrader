import sys
import os
import re
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data.futu import FutuDataProcessor, FutuServiceError
from core.data.index_constituent import IndexConstituentManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_DATE = '2020-04-28'
END_DATE = '2026-04-28'


def load_screened_stocks():
    txt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'strategies_my', 'first_board_low_open_strategy', 'screened_stocks_raw.txt'
    )
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


def check_futu_minute_data(dp, symbol, dates):
    years_needed = set()
    for d in dates:
        dt = pd.Timestamp(d)
        years_needed.add(dt.year)

    futu_dir = os.path.join(dp._data_dir, 'market', symbol)
    hfq_years = set()
    if os.path.exists(futu_dir):
        for f in os.listdir(futu_dir):
            if f.endswith('_1m.parquet'):
                yr = f.replace('_1m.parquet', '')
                try:
                    hfq_years.add(int(yr))
                except ValueError:
                    pass

    missing_hfq = years_needed - hfq_years
    return missing_hfq


def main():
    dp = FutuDataProcessor()

    if not dp.check_futu_service():
        logger.error("富途OpenD服务未开启！")
        return

    logger.info("富途OpenD服务已连接")

    stocks = load_screened_stocks()
    logger.info("从筛选结果加载 %d 只股票" % len(stocks))

    total = len(stocks)
    downloaded = 0
    cached = 0
    failed = 0
    no_data = 0

    for i, (symbol, dates) in enumerate(sorted(stocks.items())):
        missing_years = check_futu_minute_data(dp, symbol, dates)

        if not missing_years:
            cached += 1
            if (i + 1) % 20 == 0:
                logger.info("  进度: %d/%d, 缓存命中 %d, 已下载 %d, 无数据 %d, 失败 %d" % (
                    i + 1, total, cached, downloaded, no_data, failed))
            continue

        logger.info("[%d/%d] %s: 缺少后复权分钟数据年份 %s, 开始下载..." % (
            i + 1, total, symbol, sorted(missing_years)))

        try:
            df = dp.get_data(symbol, START_DATE, END_DATE, period='1m')
            if df is not None and not df.empty:
                downloaded += 1
                logger.info("  %s: 后复权分钟数据下载成功 (%d 行)" % (symbol, len(df)))
            else:
                no_data += 1
                logger.warning("  %s: 无数据返回" % symbol)
        except FutuServiceError:
            logger.error("富途OpenD服务断开！")
            break
        except ValueError as e:
            no_data += 1
            logger.warning("  %s: %s" % (symbol, e))
        except Exception as e:
            failed += 1
            logger.error("  %s: 下载失败: %s" % (symbol, e))

        import time
        time.sleep(0.5)

    logger.info("=" * 60)
    logger.info("下载完成！")
    logger.info("  缓存命中: %d" % cached)
    logger.info("  已下载: %d" % downloaded)
    logger.info("  无数据: %d" % no_data)
    logger.info("  失败: %d" % failed)

    logger.info("\n检查不复权分钟数据...")
    raw_missing = []
    for symbol, dates in sorted(stocks.items()):
        years_needed = set()
        for d in dates:
            dt = pd.Timestamp(d)
            years_needed.add(dt.year)

        raw_dir = os.path.join(dp._data_dir, 'market_raw', symbol)
        raw_years = set()
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                if f.endswith('_1m.parquet'):
                    yr = f.replace('_1m.parquet', '')
                    try:
                        raw_years.add(int(yr))
                    except ValueError:
                        pass

        missing_raw = years_needed - raw_years
        if missing_raw:
            raw_missing.append((symbol, sorted(missing_raw)))

    if raw_missing:
        logger.info("缺少不复权分钟数据的股票: %d 只" % len(raw_missing))
        for symbol, years in raw_missing:
            logger.info("  %s: 缺 %s" % (symbol, years))
    else:
        logger.info("所有股票的不复权分钟数据完整！")


if __name__ == '__main__':
    main()

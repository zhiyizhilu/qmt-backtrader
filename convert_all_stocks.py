import sys
import os
import re
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_minute_hfq import convert_minute_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

txt_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'strategies_my', 'first_board_low_open_strategy', 'screened_stocks_raw.txt'
)

stocks = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        m = re.match(r'^(\S+):\s+\d+\s+次,\s+日期:\s+(.*)', line)
        if not m:
            continue
        symbol = m.group(1)
        stocks.append(symbol)

logger.info("共 %d 只股票需要转换分钟数据" % len(stocks))

success = 0
failed = 0
skipped = 0

for i, symbol in enumerate(stocks):
    logger.info("[%d/%d] 处理 %s ..." % (i + 1, len(stocks), symbol))
    try:
        result = convert_minute_data(symbol, jq=True, force=True)
        if result and result.get('minute_count', 0) > 0:
            mc = result.get('minute_count', 0)
            qc = result.get('qmt_count', 0)
            jc = result.get('jq_count', 0)
            logger.info("  %s: 成功! 总%d条 (QMT:%d, JQ:%d)" % (symbol, mc, qc, jc))
            success += 1
        else:
            logger.warning("  %s: 无数据" % symbol)
            skipped += 1
    except Exception as e:
        logger.error("  %s: 失败 - %s" % (symbol, e))
        failed += 1

    if (i + 1) % 10 == 0:
        logger.info("  进度: %d/%d, 成功 %d, 跳过 %d, 失败 %d" % (
            i + 1, len(stocks), success, skipped, failed))

logger.info("=" * 60)
logger.info("转换完成!")
logger.info("  成功: %d" % success)
logger.info("  跳过: %d" % skipped)
logger.info("  失败: %d" % failed)

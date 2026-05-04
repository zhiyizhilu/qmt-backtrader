import argparse
import os
import logging
from api.backtest_api import BacktestAPI
from core.cache import cache_manager
from core.data.index_constituent import IndexConstituentManager
from api.qmt_api import QMTAPI
from api.instance_manager import StrategyInstanceManager
from core.stock_selection import StockSelectionStrategy
from core.virtual_book import VirtualBook
from strategies import (get_strategy, get_strategy_default_kwargs,
                        get_strategy_choices, get_strategy_backtest_config)
from utils.logger import Logger


def _setup_debug_logging():
    """设置调试日志级别 - 确保DEBUG日志输出到控制台"""
    log_level = os.environ.get('QMT_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)

    # 为关键模块设置日志级别和console handler
    key_modules = ['api.backtest_api', 'core.strategy', 'core.executor',
                   'core.data_adapter', 'core.data', 'core.strategy_logic',
                   'core.stock_selection', 'core.financial_data',
                   'strategies.example_strategy',
                   'strategies.etf_rotation_strategy',
                   'strategies.fundamental_strategy',
                   'strategies.high_dividend_strategy']

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    for module_name in key_modules:
        mod_logger = logging.getLogger(module_name)
        mod_logger.setLevel(level)
        # 避免重复添加handler
        if level <= logging.DEBUG and not any(
            isinstance(h, logging.StreamHandler) and h.level == logging.DEBUG
            for h in mod_logger.handlers
        ):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            mod_logger.addHandler(console_handler)


def _resolve_strategy(strategy_name: str):
    """根据名称解析策略类、默认参数和回测配置"""
    strategy_class = get_strategy(strategy_name)
    if strategy_class is None:
        raise ValueError(f"未知策略: {strategy_name}，可用策略: {get_strategy_choices()}")
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)
    return strategy_class, default_kwargs, backtest_config


def _init_virtual_book_from_account(api: QMTAPI, book: VirtualBook):
    """从账户实际状态初始化 VirtualBook"""
    if not api.trader:
        return
    actual_positions = {}
    actual_cash = 0.0
    try:
        positions = api.trader.get_position()
        if positions:
            for pos in positions:
                symbol = getattr(pos, 'stock_code', str(pos))
                volume = getattr(pos, 'volume', 0)
                if volume > 0:
                    actual_positions[symbol] = volume
        account = api.trader.get_account()
        if account and hasattr(account, 'cash'):
            actual_cash = account.cash
    except Exception:
        pass
    book.initialize_from_account(actual_positions, actual_cash, set())


def run_backtest(strategy_name='double_ma', period='1d', pool='沪深A股',
                 start_date=None, end_date=None, proxy='', ai_mode=False,
                 no_record=False):
    """运行回测"""
    _setup_debug_logging()
    log_file = Logger.setup_global_file_handler(strategy_name)
    logger = Logger.get_default_logger(strategy_name)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"开始回测 (周期: {period})")
    if proxy:
        logger.info(f"使用代理: {proxy}")
    if ai_mode:
        logger.info("AI自动运行模式已启用，将跳过所有图形界面渲染")

    strategy_class, default_kwargs, backtest_config = _resolve_strategy(strategy_name)

    config = dict(backtest_config)
    config['period'] = period
    if start_date:
        config['start_date'] = start_date
    if end_date:
        config['end_date'] = end_date

    benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(pool, '000300.SH')
    config.setdefault('benchmark', benchmark)

    api = BacktestAPI(proxy=proxy)
    if ai_mode:
        api.set_ai_mode(True)
    if no_record:
        api.set_no_record(True)

    api.set_strategy_name(strategy_name)
    api.set_backtest_config(config)

    if issubclass(strategy_class, StockSelectionStrategy):
        api.configure(**config)
        api.load_financial_data(sector=pool)
        api.add_stock_selection_strategy(strategy_class, **default_kwargs)
    else:
        api.configure(**config)
        api.add_strategy(strategy_class, **default_kwargs)

    results = api.run()

    if results:
        api.show_report()
    else:
        logger.info("回测未产生结果，可能是因为没有数据")

    logger.info("回测完成")


def run_sim_trade(strategy_name='double_ma', path=r'D:\qmt\userdata_mini', account_id=None):
    """运行模拟交易"""
    log_file = Logger.setup_global_file_handler(strategy_name)
    logger = Logger.get_default_logger(strategy_name)
    logger.info(f"日志文件: {log_file}")
    logger.info("开始模拟交易")

    strategy_class, default_kwargs, _ = _resolve_strategy(strategy_name)

    api = QMTAPI(is_sim=True, path=path, account_id=account_id)
    book = VirtualBook(strategy_id=strategy_name)
    api.add_strategy(strategy_class, instance_id=strategy_name, virtual_book=book, **default_kwargs)
    _init_virtual_book_from_account(api, book)

    api.run()
    api.close()

    logger.info("模拟交易完成")


def run_real_trade(strategy_name='double_ma', path=r'D:\qmt\userdata_mini', account_id=None):
    """运行实盘交易"""
    log_file = Logger.setup_global_file_handler(strategy_name)
    logger = Logger.get_default_logger(strategy_name)
    logger.info(f"日志文件: {log_file}")
    logger.info("开始实盘交易")

    strategy_class, default_kwargs, _ = _resolve_strategy(strategy_name)

    api = QMTAPI(is_sim=False, path=path, account_id=account_id)
    book = VirtualBook(strategy_id=strategy_name)
    api.add_strategy(strategy_class, instance_id=strategy_name, virtual_book=book, **default_kwargs)
    _init_virtual_book_from_account(api, book)

    api.run()
    api.close()

    logger.info("实盘交易完成")


def run_instances(config_path: str):
    """运行多策略实例模式"""
    import strategies

    log_file = Logger.setup_global_file_handler('instances')
    logger = Logger.get_default_logger('instances')
    logger.info(f"日志文件: {log_file}")
    logger.info(f"多策略实例模式，配置文件: {config_path}")

    manager = StrategyInstanceManager()
    manager.load_config(config_path)

    instances = manager.list_instances()
    logger.info(f"共加载 {len(instances)} 个策略实例:")
    for inst in instances:
        logger.info(
            f"  - {inst['instance_id']}: "
            f"{inst['strategy_name']} ({inst['mode']}), "
            f"账户={inst['account_id']}, "
            f"初始资金={inst['initial_capital']}"
        )

    try:
        manager.start_all()
        logger.info("所有策略实例已启动，按 Ctrl+C 停止")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        manager.stop_all()
        logger.info("所有策略实例已停止")


def main():
    """主函数"""
    import strategies

    parser = argparse.ArgumentParser(description='量化交易框架')
    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['backtest', 'sim', 'real', 'instances'],
                        help='运行模式: backtest=回测, sim=模拟交易, real=实盘交易, instances=多策略实例')
    parser.add_argument('--strategy', type=str, default='double_ma', choices=get_strategy_choices(), help='策略类型')
    parser.add_argument('--period', type=str, default='1d', choices=['1d', '1m', '5m', '15m', '30m', '60m', 'tick'], help='数据周期')
    parser.add_argument('--pool', type=str, default='沪深A股', help='股票池板块名称')
    parser.add_argument('--start', type=str, default=None, help='回测起始日期，如 2016-01-01')
    parser.add_argument('--end', type=str, default=None, help='回测结束日期，如 2026-04-17')
    parser.add_argument('--proxy', type=str, default='', help='代理地址，格式 host:port（已弃用，保留参数用于兼容性）')
    parser.add_argument('--qmt-path', type=str, default=r'D:\qmt\userdata_mini', help='QMT userdata_mini 路径')
    parser.add_argument('--account', type=str, default=None, help='QMT 资金账号，不传则自动获取第一个')
    parser.add_argument('--instances', type=str, default=None,
                        help='策略实例配置文件路径（JSON），用于 --mode instances 模式')

    # 缓存配置
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='指定缓存数据存储目录 (默认: 项目根目录下的 .cache 文件夹)')
    parser.add_argument('--mem-limit', type=int, default=500,
                        help='内存缓存最大对象数量限制 (默认: 500)')

    # 调试配置
    parser.add_argument('--debug', action='store_true', default=False,
                        help='启用DEBUG日志模式，输出详细调试信息')

    # AI自动运行模式
    parser.add_argument('--ai-mode', action='store_true', default=False,
                        help='启用AI自动运行模式，跳过所有图形界面渲染，适用于AI自动优化策略')

    # 回测记录
    parser.add_argument('--no-record', action='store_true', default=False,
                        help='禁用回测结果自动记录到本地文件')

    args = parser.parse_args()

    # 设置调试模式
    if args.debug:
        os.environ['QMT_LOG_LEVEL'] = 'DEBUG'

    # 配置缓存
    if args.cache_dir:
        cache_manager.configure(cache_dir=args.cache_dir, mem_limit=args.mem_limit)

    if args.mode == 'backtest':
        run_backtest(args.strategy, args.period, args.pool, args.start, args.end, args.proxy, args.ai_mode, args.no_record)
    elif args.mode == 'sim':
        run_sim_trade(args.strategy, args.qmt_path, args.account)
    elif args.mode == 'real':
        run_real_trade(args.strategy, args.qmt_path, args.account)
    elif args.mode == 'instances':
        if not args.instances:
            print('错误: --mode instances 需要指定 --instances 参数（配置文件路径）')
            return
        run_instances(args.instances)


if __name__ == '__main__':
    main()

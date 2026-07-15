import argparse
import os
import sys
import logging
from api.backtest_api import BacktestAPI
from core.cache import cache_manager
from core.data.index_constituent import IndexConstituentManager
from core.data.futu import FutuServiceError
from api.qmt_api import QMTAPI
from api.instance_manager import StrategyInstanceManager
from core.stock_selection import StockSelectionStrategy
from core.virtual_book import VirtualBook
from strategies import (get_strategy, get_strategy_default_kwargs,
                        get_strategy_choices, get_strategy_backtest_config)
from utils.logger import Logger
from utils.config import load_config


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
                   'strategies.high_dividend_strategy',
                   'strategies_for_vip',
                   'strategies_for_svip']

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
                volume = api.trader.get_position_volume(pos)
                if volume > 0:
                    actual_positions[symbol] = volume
        account = api.trader.get_account()
        if account and hasattr(account, 'cash'):
            actual_cash = account.cash
    except Exception:
        pass
    book.initialize_from_account(actual_positions, actual_cash, set())


def _parse_strategy_params(params_str: str) -> dict:
    """解析策略参数字符串，格式: key=value,key=value

    支持自动类型推断:
    - 整数: 100, -5
    - 浮点数: 0.5, -0.003, 1e-4
    - 布尔值: true/false
    - 字符串: 其他值
    """
    if not params_str:
        return {}
    result = {}
    for pair in params_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        if value.lower() == 'true':
            result[key] = True
        elif value.lower() == 'false':
            result[key] = False
        else:
            try:
                if '.' in value or 'e' in value.lower():
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def run_backtest(strategy_name='double_ma', period='1d', pool=None,
                 start_date=None, end_date=None, proxy='', ai_mode=False,
                 no_record=False, slippage=None, data_source='qmt',
                 strategy_params=None, lazy_mode=True):
    """运行回测"""
    _setup_debug_logging()
    log_file = Logger.setup_global_file_handler(strategy_name)
    logger = Logger.get_default_logger(strategy_name)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"开始回测 (周期: {period}, 数据源: {data_source})")
    if proxy:
        logger.info(f"使用代理: {proxy}")
    if ai_mode:
        logger.info("AI自动运行模式已启用，将跳过所有图形界面渲染")

    strategy_class, default_kwargs, backtest_config = _resolve_strategy(strategy_name)

    if strategy_params:
        default_kwargs.update(strategy_params)
        if 'symbol' in strategy_params:
            symbol = strategy_params['symbol']
            backtest_config = dict(backtest_config)
            backtest_config['benchmark'] = symbol
            backtest_config['compare_symbols'] = [symbol]

    config = dict(backtest_config)
    config['period'] = period
    if start_date:
        config['start_date'] = start_date
    if end_date:
        config['end_date'] = end_date
    if slippage is not None:
        config['slippage'] = slippage

    pool = pool if pool is not None else backtest_config.get('pool', '沪深A股')
    config['pool'] = pool

    if 'benchmark' not in config:
        benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(pool, '000300.SH')
        config['benchmark'] = benchmark

    try:
        import time as _time
        bt_start = _time.time()
        api = BacktestAPI(proxy=proxy, data_source=data_source)
        if ai_mode:
            api.set_ai_mode(True)
        if no_record:
            api.set_no_record(True)
        if lazy_mode:
            api.set_lazy_mode(True)

        api.set_strategy_name(strategy_name)
        api.set_log_file(log_file)
        api.set_backtest_config(config)

        if issubclass(strategy_class, StockSelectionStrategy):
            api.configure(**config)
            api.load_financial_data(sector=pool)
            api.add_stock_selection_strategy(strategy_class, **default_kwargs)
        else:
            api.configure(**config)
            api.add_strategy(strategy_class, **default_kwargs)

        results = api.run()
        bt_elapsed = _time.time() - bt_start

        if results:
            if bt_elapsed < 60:
                logger.info(f"回测耗时: {bt_elapsed:.2f}秒")
            elif bt_elapsed < 3600:
                logger.info(f"回测耗时: {bt_elapsed / 60:.2f}分钟")
            else:
                logger.info(f"回测耗时: {bt_elapsed / 3600:.2f}小时")
            api.show_report()
        else:
            logger.info("回测未产生结果，可能是因为没有数据")

        logger.info("回测完成")
    except FutuServiceError as e:
        logger.error(str(e))
        sys.exit(1)


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


def _resolve_instances_config(config_path: str) -> str:
    """解析实例配置文件路径

    支持三种格式:
    - 绝对路径: /path/to/config.json
    - 相对路径: config/instances_sim_config.json
    - 简名: sim → 自动查找 config/instances_sim_config.json → config/sim.json
    """
    if os.path.isfile(config_path):
        return config_path

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

    # 尝试在 config/ 目录下查找
    candidates = [
        os.path.join(config_dir, config_path),
        os.path.join(config_dir, f'instances_{config_path}_config.json'),
        os.path.join(config_dir, f'{config_path}.json'),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f'找不到实例配置文件: {config_path}\n'
        f'已查找: {config_path}, ' + ', '.join(candidates)
    )


def run_instances(config_path: str):
    """运行多策略实例模式"""
    import strategies

    config_path = _resolve_instances_config(config_path)

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

    import signal
    import threading

    shutdown_event = threading.Event()

    def _sigint_handler(signum, frame):
        if shutdown_event.is_set():
            return
        logger.info("收到停止信号 (Ctrl+C)")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        manager.start_all()
        # 注册 atexit 兜底保存，防止异常退出未触发 stop_all
        import atexit
        def _atexit_save():
            if manager._running:
                logger.info("atexit 触发状态保存")
                manager.stop_all()
        atexit.register(_atexit_save)
        logger.info("所有策略实例已启动，按 Ctrl+C 停止")
        shutdown_event.wait()
    except Exception as e:
        logger.error(f"运行异常: {e}")
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
    parser.add_argument('--pool', type=str, default=None, help='股票池板块名称（不指定则使用策略默认值）')
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
    parser.add_argument('--slippage', type=float, default=None,
                        help='滑点百分比, 如0.001表示0.1%%, 不传则使用策略默认值')
    parser.add_argument('--no-lazy', action='store_false', dest='lazy',
                        help='禁用按需加载模式，预加载所有标的数据（默认启用按需加载）')
    parser.add_argument('--data-source', type=str, default='qmt',
                        choices=['qmt', 'open', 'futu'],
                        help='行情数据源: qmt=QMT(默认), open=OpenData, futu=富途本地数据')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径，配置覆盖默认值，命令行参数覆盖 YAML 配置')
    parser.add_argument('--strategy-params', type=str, default=None,
                        help='策略参数覆盖, 格式: key=value,key=value, 如 symbol=000001.SZ,base_position_ratio=0.3')

    args = parser.parse_args()

    yaml_config = load_config(args.config)

    if args.debug:
        os.environ['QMT_LOG_LEVEL'] = 'DEBUG'

    if not args.cache_dir:
        args.cache_dir = yaml_config.get('cache', {}).get('dir')
    if not args.mem_limit or args.mem_limit == 500:
        yaml_mem = yaml_config.get('cache', {}).get('mem_limit')
        if yaml_mem and args.mem_limit == 500:
            args.mem_limit = yaml_mem

    if args.cache_dir:
        cache_manager.configure(cache_dir=args.cache_dir, mem_limit=args.mem_limit)

    if args.mode == 'backtest':
        sp = _parse_strategy_params(args.strategy_params) if args.strategy_params else None
        run_backtest(args.strategy, args.period, args.pool, args.start, args.end, args.proxy, args.ai_mode, args.no_record, args.slippage, args.data_source, sp, args.lazy)
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

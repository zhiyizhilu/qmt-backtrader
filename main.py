import argparse
from api.backtest_api import BacktestAPI
from api.qmt_api import QMTAPI
from core.stock_selection import StockSelectionStrategy
from strategies import (get_strategy, get_strategy_default_kwargs,
                        get_strategy_choices, get_strategy_backtest_config)
from utils.logger import Logger


def _resolve_strategy(strategy_name: str):
    """根据名称解析策略类、默认参数和回测配置"""
    strategy_class = get_strategy(strategy_name)
    if strategy_class is None:
        raise ValueError(f"未知策略: {strategy_name}，可用策略: {get_strategy_choices()}")
    default_kwargs = get_strategy_default_kwargs(strategy_name)
    backtest_config = get_strategy_backtest_config(strategy_name)
    return strategy_class, default_kwargs, backtest_config


def run_backtest(strategy_name='double_ma', period='1d', pool='沪深A股',
                 start_date=None, end_date=None, data_source='qmt'):
    """运行回测"""
    logger = Logger.get_default_logger(strategy_name)
    logger.info(f"开始回测 (周期: {period}, 数据源: {data_source})")

    strategy_class, default_kwargs, backtest_config = _resolve_strategy(strategy_name)

    config = dict(backtest_config)
    config['period'] = period
    if start_date:
        config['start_date'] = start_date
    if end_date:
        config['end_date'] = end_date

    api = BacktestAPI(data_source=data_source)

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


def run_sim_trade(strategy_name='double_ma'):
    """运行模拟交易"""
    logger = Logger.get_default_logger(strategy_name)
    logger.info("开始模拟交易")

    strategy_class, default_kwargs, _ = _resolve_strategy(strategy_name)

    api = QMTAPI(is_sim=True)
    api.add_strategy(strategy_class, **default_kwargs)
    api.run()
    api.close()

    logger.info("模拟交易完成")


def run_real_trade(strategy_name='double_ma'):
    """运行实盘交易"""
    logger = Logger.get_default_logger(strategy_name)
    logger.info("开始实盘交易")

    strategy_class, default_kwargs, _ = _resolve_strategy(strategy_name)

    api = QMTAPI(is_sim=False)
    api.add_strategy(strategy_class, **default_kwargs)
    api.run()
    api.close()

    logger.info("实盘交易完成")


def main():
    """主函数"""
    import strategies

    parser = argparse.ArgumentParser(description='量化交易框架')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'sim', 'real'], help='运行模式')
    parser.add_argument('--strategy', type=str, default='double_ma', choices=get_strategy_choices(), help='策略类型')
    parser.add_argument('--period', type=str, default='1d', choices=['1d', '1m', '5m', '15m', '30m', '60m', 'tick'], help='数据周期')
    parser.add_argument('--pool', type=str, default='沪深A股', help='股票池板块名称')
    parser.add_argument('--start', type=str, default=None, help='回测起始日期，如 2016-01-01')
    parser.add_argument('--end', type=str, default=None, help='回测结束日期，如 2026-04-17')
    parser.add_argument('--data-source', type=str, default='qmt', choices=['qmt', 'akshare', 'baostock'], help='数据源: qmt(需QMT客户端), akshare(免费在线), baostock(免费在线)')

    args = parser.parse_args()

    if args.mode == 'backtest':
        run_backtest(args.strategy, args.period, args.pool, args.start, args.end, args.data_source)
    elif args.mode == 'sim':
        run_sim_trade(args.strategy)
    elif args.mode == 'real':
        run_real_trade(args.strategy)


if __name__ == '__main__':
    main()

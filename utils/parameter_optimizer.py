import itertools
import logging
from typing import Dict, List, Optional, Any, Tuple, Type
from core.strategy_logic import StrategyLogic
from core.data import QMTDataProcessor


class ParameterOptimizer:
    """参数优化工具 - 通过BacktestAPI运行回测，与框架解耦"""

    @staticmethod
    def grid_search(
        strategy_logic_class: Type[StrategyLogic],
        param_grid: Dict[str, List[Any]],
        symbols: List[str],
        start_date: str,
        end_date: str,
        period: str = '1d',
        initial_cash: float = 200000,
        commission: float = 0.0001,
        **kwargs,
    ):
        """网格搜索优化参数

        Args:
            strategy_logic_class: 策略逻辑类（继承StrategyLogic）
            param_grid: 参数网格，如 {'fast_period': [5, 10], 'slow_period': [20, 30]}
            symbols: 标的列表
            start_date: 回测起始日期
            end_date: 回测结束日期
            period: 数据周期
            initial_cash: 初始资金
            commission: 佣金费率
        """
        from api.backtest_api import BacktestAPI
        import datetime

        logger = logging.getLogger(__name__)
        best_score = -float('inf')
        best_params = {}

        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        data_start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=40)
        data_start_date = data_start_dt.strftime('%Y-%m-%d')

        data_processor = QMTDataProcessor()
        data_cache: Dict[str, Any] = {}
        for symbol in symbols:
            try:
                data_cache[symbol] = data_processor.get_data(symbol, data_start_date, end_date, period)
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")

        for idx, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))

            try:
                import backtrader as bt

                cerebro = bt.Cerebro()
                cerebro.broker.setcash(initial_cash)
                cerebro.broker.setcommission(commission=commission)
                cerebro.broker.set_checksubmit(False)

                for symbol, data in data_cache.items():
                    if data is not None and not data.empty:
                        bt_data = bt.feeds.PandasData(
                            dataname=data,
                            datetime='datetime' if 'datetime' in data.columns else None,
                            open='open', high='high', low='low', close='close',
                            volume='volume',
                            openinterest='openinterest' if 'openinterest' in data.columns else -1,
                        )
                        cerebro.adddata(bt_data, name=symbol)

                from api.backtest_api import BacktestStrategyAdapter
                from core.executor import BacktestExecutor
                from core.data_adapter import BacktraderDataAdapter

                class _OptAdapter(BacktestStrategyAdapter):
                    def __init__(self, **bt_kwargs):
                        super().__init__(
                            strategy_logic_class=strategy_logic_class,
                            strategy_kwargs=param_dict,
                            symbols=symbols,
                            period=period,
                            **bt_kwargs,
                        )

                cerebro.addstrategy(_OptAdapter, trade_start_date=start_date)
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

                results = cerebro.run()
                if results:
                    final_value = cerebro.broker.getvalue()
                    score = (final_value - initial_cash) / initial_cash

                    if score > best_score:
                        best_score = score
                        best_params = param_dict
                        logger.info(f"[{idx + 1}/{len(param_combinations)}] New best: {best_params}, score: {score:.4f}")
                    else:
                        logger.debug(f"[{idx + 1}/{len(param_combinations)}] {param_dict}, score: {score:.4f}")

            except Exception as e:
                logger.error(f"参数组合 {param_dict} 回测失败: {e}")

        return best_params, best_score

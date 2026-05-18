import itertools
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Callable, Type
from core.strategy_logic import StrategyLogic
from core.data import QMTDataProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _grid_search_worker(args):
    """模块级函数：单次回测执行，供 ProcessPoolExecutor 调用

    Args:
        args: 元组 (strategy_logic_class, param_dict, symbols, start_date, end_date,
              period, initial_cash, commission, slippage, score_fn)

    Returns:
        (param_dict, score) 元组
    """
    (strategy_logic_class, param_dict, symbols, start_date, end_date,
     period, initial_cash, commission, slippage, score_fn) = args

    from api.backtest_api import BacktestAPI
    from utils.report import set_ai_mode
    import datetime as dt

    set_ai_mode(True)

    api = BacktestAPI()
    api.set_ai_mode(True)
    api.set_no_record(True)
    api.configure(
        cash=initial_cash,
        commission=commission,
        slippage=slippage,
        start_date=start_date,
        end_date=end_date,
        period=period,
        trade_start_date=start_date,
    )

    data_start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d') - dt.timedelta(days=40)
    data_start_date = data_start_dt.strftime('%Y-%m-%d')

    for symbol in symbols:
        try:
            api.add_data(symbol, data_start_date, end_date, period)
        except Exception:
            pass

    try:
        api.add_strategy(strategy_logic_class, **param_dict)
    except Exception:
        pass

    api.run()
    result = api.get_result()

    if result is not None:
        if score_fn is not None:
            score = score_fn(result)
        else:
            score = result.account.rate
    else:
        score = -float('inf')

    return param_dict, score


class ParameterOptimizer:
    """参数优化工具 - 通过BacktestAPI运行回测，与框架解耦"""

    @staticmethod
    def walk_forward(
        strategy_logic_class: Type[StrategyLogic],
        start_date: str,
        end_date: str,
        train_months: int = 60,
        test_months: int = 12,
        step_months: int = 12,
        anchor: bool = False,
        **backtest_kwargs,
    ):
        """Walk-Forward 验证

        Args:
            strategy_logic_class: 策略逻辑类（继承StrategyLogic）
            start_date: 总起始日期
            end_date: 总结束日期
            train_months: 训练窗口月数
            test_months: 测试窗口月数
            step_months: 滑动步长月数
            anchor: False=滚动模式, True=锚定模式
            **backtest_kwargs: 传递给 WalkForwardValidator 的回测参数，
                支持 config(回测配置dict), strategy_kwargs(策略参数dict),
                pool(股票池), is_stock_selection(是否选股策略)
        """
        from utils.walk_forward import WalkForwardSplitter, WalkForwardValidator

        splitter = WalkForwardSplitter(
            total_start=start_date,
            total_end=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            anchor=anchor,
        )
        validator = WalkForwardValidator(
            strategy_logic_class=strategy_logic_class,
            splitter=splitter,
            **backtest_kwargs,
        )
        validator.run()
        return validator.summary()

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
        slippage: float = 0.0,
        n_workers: int = 1,
        score_fn: Optional[Callable] = None,
        output_file: Optional[str] = None,
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
            slippage: 滑点百分比，如0.001表示0.1%，0表示无滑点
            n_workers: 并行工作进程数，1为顺序执行
            score_fn: 自定义评分函数，接收BacktestingResult返回float，None时使用总收益率
            output_file: 结果持久化JSON文件路径，None时不保存
        """
        from utils.report import set_ai_mode
        set_ai_mode(True)

        from api.backtest_api import BacktestAPI
        import datetime as dt

        logger = logging.getLogger(__name__)
        best_score = -float('inf')
        best_params = {}

        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        data_start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d') - dt.timedelta(days=40)
        data_start_date = data_start_dt.strftime('%Y-%m-%d')

        all_results = []

        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            task_args_list = []
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                task_args_list.append((
                    strategy_logic_class, param_dict, symbols, start_date, end_date,
                    period, initial_cash, commission, slippage, score_fn,
                ))

            if tqdm is not None:
                pbar = tqdm(total=len(task_args_list), desc="Grid Search")
            else:
                pbar = None

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_grid_search_worker, args): args[1]
                    for args in task_args_list
                }
                for future in as_completed(futures):
                    param_dict = futures[future]
                    try:
                        _, score = future.result()
                        all_results.append({'params': param_dict, 'score': score})
                        if score > best_score:
                            best_score = score
                            best_params = param_dict
                            logger.info(f"New best: {best_params}, score: {score:.4f}")
                        else:
                            logger.debug(f"{param_dict}, score: {score:.4f}")
                    except Exception as e:
                        logger.error(f"参数组合 {param_dict} 回测失败: {e}")
                        all_results.append({'params': param_dict, 'score': None, 'error': str(e)})
                    if pbar is not None:
                        pbar.update(1)
                    else:
                        print(f"[{len(all_results)}/{len(param_combinations)}] 完成")

            if pbar is not None:
                pbar.close()

        else:
            data_processor = QMTDataProcessor()
            data_cache: Dict[str, Any] = {}
            for symbol in symbols:
                try:
                    data_cache[symbol] = data_processor.get_data(symbol, data_start_date, end_date, period)
                except Exception as e:
                    logger.error(f"获取 {symbol} 数据失败: {e}")

            if tqdm is not None:
                iterator = tqdm(enumerate(param_combinations), total=len(param_combinations), desc="Grid Search")
            else:
                iterator = enumerate(param_combinations)

            for idx, params in iterator:
                param_dict = dict(zip(param_names, params))

                try:
                    api = BacktestAPI()
                    api.set_ai_mode(True)
                    api.set_no_record(True)
                    api.configure(
                        cash=initial_cash,
                        commission=commission,
                        slippage=slippage,
                        start_date=start_date,
                        end_date=end_date,
                        period=period,
                        trade_start_date=start_date,
                    )

                    for symbol, data in data_cache.items():
                        if data is not None and not data.empty:
                            api._engine.add_data(symbol, data)
                            api._symbols.append(symbol)
                            api._data_cache[symbol] = data

                    api._data_start_date = data_start_date
                    api._data_end_date = end_date
                    api._period = period

                    api.add_strategy(strategy_logic_class, **param_dict)
                    api.run()
                    result = api.get_result()

                    if result is not None:
                        if score_fn is not None:
                            score = score_fn(result)
                        else:
                            score = result.account.rate
                    else:
                        score = -float('inf')

                    all_results.append({'params': param_dict, 'score': score})

                    if score > best_score:
                        best_score = score
                        best_params = param_dict
                        logger.info(f"[{idx + 1}/{len(param_combinations)}] New best: {best_params}, score: {score:.4f}")
                    else:
                        logger.debug(f"[{idx + 1}/{len(param_combinations)}] {param_dict}, score: {score:.4f}")

                except Exception as e:
                    logger.error(f"参数组合 {param_dict} 回测失败: {e}")
                    all_results.append({'params': param_dict, 'score': None, 'error': str(e)})

        if output_file is not None:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"结果已保存到 {output_file}")
            except Exception as e:
                logger.error(f"结果保存失败: {e}")

        set_ai_mode(False)
        return best_params, best_score

import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple, Type

from dateutil.relativedelta import relativedelta

from core.strategy_logic import StrategyLogic


class WalkForwardSplitter:
    """Walk-Forward 时间窗口分割器

    滚动模式(anchor=False): 训练窗口固定长度向前滑动
    锚定模式(anchor=True): 训练窗口起点固定，终点向前扩展
    """

    def __init__(
        self,
        total_start: str,
        total_end: str,
        train_months: int = 60,
        test_months: int = 12,
        step_months: int = 12,
        anchor: bool = False,
    ):
        self.total_start = total_start
        self.total_end = total_end
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.anchor = anchor

    def split(self) -> List[Tuple[str, str, str, str]]:
        """生成 (train_start, train_end, test_start, test_end) 列表"""
        start_dt = datetime.datetime.strptime(self.total_start, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(self.total_end, '%Y-%m-%d')

        splits: List[Tuple[str, str, str, str]] = []
        cursor = start_dt + relativedelta(months=self.train_months)

        while True:
            train_end_dt = cursor
            test_start_dt = cursor
            test_end_dt = cursor + relativedelta(months=self.test_months)

            if test_end_dt > end_dt:
                break

            if self.anchor:
                train_start_dt = start_dt
            else:
                train_start_dt = cursor - relativedelta(months=self.train_months)

            ts = train_start_dt.strftime('%Y-%m-%d')
            te = train_end_dt.strftime('%Y-%m-%d')
            oos_s = test_start_dt.strftime('%Y-%m-%d')
            oos_e = test_end_dt.strftime('%Y-%m-%d')

            splits.append((ts, te, oos_s, oos_e))

            cursor = cursor + relativedelta(months=self.step_months)

        return splits


class WalkForwardValidator:
    """Walk-Forward 验证器

    对每个 split 执行 IS(样本内) 和 OOS(样本外) 回测，
    收集指标并计算退化率。
    """

    def __init__(
        self,
        strategy_logic_class: Type[StrategyLogic],
        splitter: WalkForwardSplitter,
        **backtest_kwargs,
    ):
        self.strategy_logic_class = strategy_logic_class
        self.splitter = splitter
        self.backtest_kwargs = backtest_kwargs
        self._results: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_metrics(result) -> Dict[str, Any]:
        """从 BacktestingResult 提取关键指标"""
        metrics: Dict[str, Any] = {}
        if result is None:
            metrics['sharpe_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
            metrics['total_return'] = 0.0
            metrics['annual_return'] = 0.0
            metrics['trading_days'] = 0
            return metrics

        sr = result.sharpe_ratio()
        dd = result.max_drawdown()
        acc = result.account
        metrics['sharpe_ratio'] = sr
        metrics['max_drawdown'] = dd
        metrics['total_return'] = acc.rate
        metrics['initial_capital'] = acc.initial_capital
        metrics['final_value'] = acc.dynamic_rights

        if result.df is not None and len(result.df) > 0:
            days = len(result.df)
            years = days / 252
            annual_ret = (1 + acc.rate) ** (1 / years) - 1 if years > 0 else 0
            if isinstance(annual_ret, complex):
                annual_ret = annual_ret.real
            metrics['annual_return'] = float(annual_ret)
            metrics['trading_days'] = days
        else:
            metrics['annual_return'] = 0.0
            metrics['trading_days'] = 0

        metrics['fee'] = getattr(result, 'total_fee', 0)
        metrics['turnover'] = getattr(result, 'turnover', 0)
        return metrics

    def _run_single_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """执行单次回测并返回指标"""
        from api.backtest_api import BacktestAPI

        config = dict(self.backtest_kwargs.get('config', {}))
        config['start_date'] = start_date
        config['end_date'] = end_date

        strategy_kwargs = dict(self.backtest_kwargs.get('strategy_kwargs', {}))
        pool = self.backtest_kwargs.get('pool', None)
        is_stock_selection = self.backtest_kwargs.get('is_stock_selection', False)

        api = BacktestAPI()
        api.set_ai_mode(True)
        api.set_no_record(True)
        api.configure(**config)

        if pool:
            api.load_financial_data(sector=pool)

        if is_stock_selection:
            api.add_stock_selection_strategy(self.strategy_logic_class, **strategy_kwargs)
        else:
            api.add_strategy(self.strategy_logic_class, **strategy_kwargs)

        api.run()
        result = api.get_result()
        return self._extract_metrics(result)

    def run(self) -> List[Dict[str, Any]]:
        """对每个 split 执行 IS/OOS 回测"""
        logger = logging.getLogger(__name__)
        splits = self.splitter.split()
        self._results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(
                f"Walk-Forward 第{i + 1}/{len(splits)}轮: "
                f"IS={train_start}~{train_end}, OOS={test_start}~{test_end}"
            )

            is_metrics = self._run_single_backtest(train_start, train_end)
            oos_metrics = self._run_single_backtest(test_start, test_end)

            is_sharpe = is_metrics.get('sharpe_ratio', 0.0)
            oos_sharpe = oos_metrics.get('sharpe_ratio', 0.0)
            decay_ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

            round_result = {
                'round': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'is': is_metrics,
                'oos': oos_metrics,
                'decay_ratio': decay_ratio,
            }
            self._results.append(round_result)

            logger.info(
                f"  IS夏普={is_sharpe:.4f}, OOS夏普={oos_sharpe:.4f}, 退化率={decay_ratio:.4f}"
            )

        return self._results

    def summary(self) -> Dict[str, Any]:
        """返回汇总报告"""
        if not self._results:
            return {'error': '尚未执行 run()，无结果可汇总'}

        rounds_info = []
        is_sharpes = []
        oos_sharpes = []
        decay_ratios = []

        for r in self._results:
            is_sharpe = r['is'].get('sharpe_ratio', 0.0)
            oos_sharpe = r['oos'].get('sharpe_ratio', 0.0)
            decay = r['decay_ratio']

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
            decay_ratios.append(decay)

            rounds_info.append({
                'round': r['round'],
                'train_period': f"{r['train_start']}~{r['train_end']}",
                'test_period': f"{r['test_start']}~{r['test_end']}",
                'is_sharpe': is_sharpe,
                'is_return': r['is'].get('total_return', 0.0),
                'is_max_drawdown': r['is'].get('max_drawdown', 0.0),
                'oos_sharpe': oos_sharpe,
                'oos_return': r['oos'].get('total_return', 0.0),
                'oos_max_drawdown': r['oos'].get('max_drawdown', 0.0),
                'decay_ratio': decay,
            })

        import numpy as np

        avg_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        avg_decay = float(np.mean(decay_ratios)) if decay_ratios else 0.0
        overall_decay = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0.0

        positive_decay_count = sum(1 for d in decay_ratios if d > 0)
        consistency = positive_decay_count / len(decay_ratios) if decay_ratios else 0.0

        return {
            'total_rounds': len(self._results),
            'anchor_mode': self.splitter.anchor,
            'train_months': self.splitter.train_months,
            'test_months': self.splitter.test_months,
            'step_months': self.splitter.step_months,
            'avg_is_sharpe': avg_is_sharpe,
            'avg_oos_sharpe': avg_oos_sharpe,
            'avg_decay_ratio': avg_decay,
            'overall_decay_ratio': overall_decay,
            'consistency': consistency,
            'rounds': rounds_info,
        }

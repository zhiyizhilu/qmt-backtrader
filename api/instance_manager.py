import json
import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from api.qmt_api import QMTAPI
from core.virtual_book import VirtualBook
from core.reconciler import Reconciler, ReconcileResult
from strategies import get_strategy, get_strategy_default_kwargs
from utils.logger import Logger


class StrategyInstanceManager:
    """策略实例管理器 - 管理多个策略实例的启动、运行和停止

    支持从 JSON 配置文件加载多个策略实例，按账户分组初始化，
    自动创建 VirtualBook 实现策略级隔离，并协调对账。
    """

    def __init__(self):
        self._configs: List[dict] = []
        self._apis: Dict[str, QMTAPI] = {}
        self._books: Dict[str, VirtualBook] = {}
        self._reconcilers: Dict[str, Reconciler] = {}
        self._running = False
        self._reconcile_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def load_config(self, config_path: str):
        """加载策略实例配置文件

        Args:
            config_path: JSON 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        instances = config.get('instances', [])
        if not instances:
            raise ValueError(f'配置文件中没有策略实例: {config_path}')

        instance_ids = set()
        for inst in instances:
            instance_id = inst.get('instance_id')
            if not instance_id:
                raise ValueError(f'策略实例缺少 instance_id: {inst}')
            if instance_id in instance_ids:
                raise ValueError(f'重复的 instance_id: {instance_id}')
            instance_ids.add(instance_id)

            strategy_name = inst.get('strategy_name')
            if not strategy_name:
                raise ValueError(f'策略实例缺少 strategy_name: {instance_id}')

            strategy_class = get_strategy(strategy_name)
            if strategy_class is None:
                raise ValueError(f'未知策略: {strategy_name} (实例: {instance_id})')

        self._configs = instances
        self.logger.info(f'加载配置完成: {len(instances)} 个策略实例')

    def start_all(self):
        """启动所有策略实例

        流程：
        1. 按账户分组
        2. 每个账户初始化一个 QMTAPI
        3. 查询账户实际状态
        4. 按顺序初始化 VirtualBook
        5. 校验簿记总和
        6. 创建策略实例
        7. 启动策略运行
        """
        account_groups = self._group_by_account()

        for account_key, instances in account_groups.items():
            first_config = instances[0]
            is_sim = first_config.get('mode', 'sim') == 'sim'
            account_id = first_config.get('account_id')
            qmt_path = first_config.get('qmt_path', r'D:\qmt\userdata_mini')

            self.logger.info(f'初始化账户: {account_key} (模式={"模拟" if is_sim else "实盘"})')

            api = QMTAPI(is_sim=is_sim, path=qmt_path, account_id=account_id)
            self._apis[account_key] = api

            actual_positions = self._query_positions(api)
            actual_cash = self._query_cash(api)

            self.logger.info(
                f'账户 {account_key} 实际状态: '
                f'持仓={len(actual_positions)}只, 现金={actual_cash:.2f}'
            )

            claimed_symbols = set()
            for config in instances:
                instance_id = config['instance_id']
                initial_capital = config.get('initial_capital', 0)

                book = VirtualBook(
                    strategy_id=instance_id,
                    initial_capital=initial_capital
                )

                if config.get('claim_existing_positions', True):
                    book.initialize_from_account(
                        actual_positions, actual_cash, claimed_symbols
                    )
                    claimed_symbols.update(book._positions.keys())

                self._books[instance_id] = book
                self.logger.info(
                    f'VirtualBook 初始化: {instance_id}, '
                    f'持仓={len(book._positions)}只, 现金={book.get_cash():.2f}'
                )

            self._validate_books(account_key, actual_positions, actual_cash)

            for config in instances:
                instance_id = config['instance_id']
                strategy_name = config['strategy_name']
                book = self._books[instance_id]

                strategy_class = get_strategy(strategy_name)
                kwargs = dict(get_strategy_default_kwargs(strategy_name))
                kwargs.update(config.get('kwargs', {}))

                api.add_strategy(
                    strategy_class,
                    instance_id=instance_id,
                    virtual_book=book,
                    **kwargs
                )
                self.logger.info(f'策略实例已创建: {instance_id} ({strategy_name})')

            books_for_account = [
                self._books[c['instance_id']] for c in instances
            ]
            self._reconcilers[account_key] = Reconciler(books_for_account, api.trader)

        for account_key, api in self._apis.items():
            api.run_loop()
            self.logger.info(f'策略运行已启动: {account_key}')

        self._running = True
        self._start_reconcile_timer()

    def stop_all(self):
        """停止所有策略实例"""
        self._running = False
        if self._reconcile_thread and self._reconcile_thread.is_alive():
            self._reconcile_thread.join(timeout=10)
        for account_key, api in self._apis.items():
            api.close()
            self.logger.info(f'策略已停止: {account_key}')

    def reconcile_all(self) -> Dict[str, ReconcileResult]:
        """对所有账户执行对账

        Returns:
            账户ID → 对账结果的映射
        """
        results = {}
        for account_key, reconciler in self._reconcilers.items():
            result = reconciler.reconcile()
            if not result.is_clean:
                self.logger.warning(f'账户 {account_key} 对账发现偏差:\n{result}')
                reconciler.auto_correct(result)
            else:
                self.logger.debug(f'账户 {account_key} 对账一致 ✅')
            results[account_key] = result
        return results

    def list_instances(self) -> List[dict]:
        """列出所有策略实例及其状态"""
        result = []
        for config in self._configs:
            instance_id = config['instance_id']
            book = self._books.get(instance_id)
            result.append({
                'instance_id': instance_id,
                'strategy_name': config.get('strategy_name'),
                'mode': config.get('mode'),
                'account_id': config.get('account_id'),
                'initial_capital': config.get('initial_capital', 0),
                'current_positions': len(book._positions) if book else 0,
                'current_cash': book.get_cash() if book else 0,
            })
        return result

    def get_virtual_book(self, instance_id: str) -> Optional[VirtualBook]:
        """获取指定实例的 VirtualBook"""
        return self._books.get(instance_id)

    def _group_by_account(self) -> Dict[str, List[dict]]:
        """按账户分组策略实例配置"""
        groups: Dict[str, List[dict]] = defaultdict(list)
        for config in self._configs:
            account_id = config.get('account_id', 'default')
            mode = config.get('mode', 'sim')
            key = f"{account_id}_{mode}"
            groups[key].append(config)
        return dict(groups)

    def _query_positions(self, api: QMTAPI) -> Dict[str, int]:
        """查询账户实际持仓"""
        result: Dict[str, int] = {}
        if not api.trader:
            return result
        try:
            positions = api.trader.get_position()
            if positions:
                for pos in positions:
                    symbol = getattr(pos, 'stock_code', str(pos))
                    volume = getattr(pos, 'volume', 0)
                    if volume > 0:
                        result[symbol] = volume
        except Exception as e:
            self.logger.error(f'查询账户持仓失败: {e}')
        return result

    def _query_cash(self, api: QMTAPI) -> float:
        """查询账户实际现金"""
        if not api.trader:
            return 0.0
        try:
            account = api.trader.get_account()
            if account and hasattr(account, 'cash'):
                return account.cash
        except Exception as e:
            self.logger.error(f'查询账户现金失败: {e}')
        return 0.0

    def _validate_books(self, account_key: str, actual_positions: Dict[str, int], actual_cash: float):
        """校验所有 VirtualBook 之和不超过账户实际"""
        aggregated_positions: Dict[str, int] = {}
        aggregated_cash = 0.0
        for config in self._configs:
            account_id = config.get('account_id', 'default')
            mode = config.get('mode', 'sim')
            key = f"{account_id}_{mode}"
            if key != account_key:
                continue
            book = self._books.get(config['instance_id'])
            if book:
                for symbol, volume in book._positions.items():
                    aggregated_positions[symbol] = aggregated_positions.get(symbol, 0) + volume
                aggregated_cash += book._cash

        for symbol, agg_vol in aggregated_positions.items():
            actual_vol = actual_positions.get(symbol, 0)
            if agg_vol > actual_vol:
                self.logger.error(
                    f'校验失败: {symbol} 簿记合计={agg_vol} > 实际={actual_vol}'
                )
                raise ValueError(
                    f'VirtualBook 校验失败: {symbol} 簿记合计({agg_vol}) '
                    f'超过账户实际持仓({actual_vol})'
                )

        if aggregated_cash > actual_cash + 1.0:
            self.logger.error(
                f'校验失败: 现金簿记合计={aggregated_cash:.2f} > 实际={actual_cash:.2f}'
            )
            raise ValueError(
                f'VirtualBook 校验失败: 现金簿记合计({aggregated_cash:.2f}) '
                f'超过账户实际现金({actual_cash:.2f})'
            )

        self.logger.info(f'账户 {account_key} VirtualBook 校验通过 ✅')

    def _start_reconcile_timer(self):
        """启动定时对账线程（每日开盘前对账）"""
        def _reconcile_loop():
            last_date = None
            while self._running:
                now = time.localtime()
                current_date = f'{now.tm_year}-{now.tm_mon:02d}-{now.tm_mday:02d}'

                if current_date != last_date and now.tm_hour == 9 and now.tm_min < 30:
                    self.logger.info('定时对账: 每日开盘前')
                    try:
                        self.reconcile_all()
                    except Exception as e:
                        self.logger.error(f'定时对账异常: {e}')
                    last_date = current_date

                time.sleep(60)

        self._reconcile_thread = threading.Thread(target=_reconcile_loop, daemon=True)
        self._reconcile_thread.start()
        self.logger.info('定时对账线程已启动')

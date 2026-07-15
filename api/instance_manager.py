import json
import os
import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from api.qmt_api import QMTAPI
from core.virtual_book import VirtualBook
from core.reconciler import Reconciler, ReconcileResult
from core.record_manager import RecordManager
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
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._instance_meta: Dict[str, dict] = {}
        self._record_managers: Dict[str, RecordManager] = {}
        self._heartbeat_interval: int = 60
        self._max_restart_attempts: int = 3
        self._restart_counts: Dict[str, int] = defaultdict(int)
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
            trade_mode = first_config.get('trade_mode', 'miniqmt')
            bridge_url = first_config.get('bridge_url', 'http://127.0.0.1:8888')

            self.logger.info(f'初始化账户: {account_key} (模式={"模拟" if is_sim else "实盘"}, 交易={trade_mode})')

            api = QMTAPI(is_sim=is_sim, path=qmt_path, account_id=account_id,
                         trade_mode=trade_mode, bridge_url=bridge_url)
            self._apis[account_key] = api

            actual_positions = self._query_positions(api)
            actual_cash = self._query_cash(api)

            self.logger.info(
                f'账户 {account_key} 实际状态: '
                f'持仓={len(actual_positions)}只, 现金={actual_cash:.2f}'
            )

            claimed_symbols = set()
            persisted_states: Dict[str, dict] = {}
            for config in instances:
                instance_id = config['instance_id']
                initial_capital = config.get('initial_capital', 0)
                cash_ratio = config.get('cash_ratio', 1.0)

                book = VirtualBook(
                    strategy_id=instance_id,
                    initial_capital=initial_capital,
                    cash_ratio=cash_ratio
                )

                # 创建 RecordManager 并尝试加载持久化状态
                rm = RecordManager(instance_id)
                self._record_managers[instance_id] = rm
                persisted_state = rm.load_state()
                state_loaded = False
                if persisted_state:
                    book.set_state(persisted_state.get('virtual_book', {}))
                    persisted_states[instance_id] = persisted_state
                    state_loaded = True

                if config.get('claim_existing_positions', True):
                    book.initialize_from_account(
                        actual_positions, actual_cash, claimed_symbols,
                        cash_ratio=cash_ratio
                    )
                    claimed_symbols.update(book._positions.keys())
                else:
                    # 不认领持仓的策略：若已加载持久化状态则跳过虚拟资金初始化
                    if not state_loaded:
                        # initial_capital 直接作为虚拟现金
                        # 不受账户实际现金约束（模拟/虚拟簿记场景）
                        if initial_capital > 0:
                            book._cash = initial_capital
                        elif cash_ratio < 1.0:
                            book._cash = actual_cash * cash_ratio
                        book._last_sync_time = time.time()

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
                # 恢复策略逻辑状态（如 _last_trade_date 等防重复字段）
                if instance_id in persisted_states:
                    strategy = api._strategies.get(instance_id)
                    if strategy:
                        strategy.set_state(
                            persisted_states[instance_id].get('strategy', {})
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
        self._register_all_instances()
        self._start_heartbeat()
        self._register_trade_callbacks()

    def _register_trade_callbacks(self):
        """为每个账户的 QMTAPI 注册成交后状态保存回调"""
        for account_key, api in self._apis.items():
            def _make_callback():
                # 闭包捕获 manager 引用，instance_id 由回调参数传入
                def _callback(instance_id: str, trade_info):
                    self._save_instance_state(instance_id, trade_info)
                return _callback
            api.set_on_trade_filled_callback(_make_callback())
        self.logger.info('已为所有账户注册成交状态保存回调')

    def _save_instance_state(self, instance_id: str, trade_info):
        """保存指定实例的状态

        Args:
            instance_id: 策略实例ID
            trade_info: 成交信息，None 时跳过成交记录追加
        """
        rm = self._record_managers.get(instance_id)
        book = self._books.get(instance_id)
        if not rm or not book:
            return
        # 查找策略实例（遍历所有 api 的 _strategies）
        strategy = None
        for api in self._apis.values():
            strategy = api._strategies.get(instance_id)
            if strategy:
                break
        strategy_state = strategy.get_state() if strategy else {}
        rm.save_state(book.get_state(), strategy_state)
        if trade_info is not None:
            import datetime as _dt
            rm.append_trade({
                'time': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'instance_id': instance_id,
                'symbol': trade_info.symbol,
                'direction': trade_info.direction,
                'price': trade_info.price,
                'volume': trade_info.volume,
                'commission': trade_info.commission,
                'order_id': trade_info.order_id,
            })

    def stop_all(self):
        self._running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=10)
        if self._reconcile_thread and self._reconcile_thread.is_alive():
            self._reconcile_thread.join(timeout=10)
        # 退出前保存所有实例状态
        for instance_id in list(self._record_managers.keys()):
            self._save_instance_state(instance_id, None)
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
                    volume = api.trader.get_position_volume(pos)
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
        virtual_cash_instances = []
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
                # 跟踪哪些实例使用了虚拟资金（initial_capital > 0 且不认领持仓）
                if config.get('initial_capital', 0) > 0 and not config.get('claim_existing_positions', False):
                    virtual_cash_instances.append((config['instance_id'], book._cash))

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

        # 现金校验：使用虚拟资金的实例不纳入合计
        virtual_cash_total = sum(c for _, c in virtual_cash_instances)
        non_virtual_cash = aggregated_cash - virtual_cash_total
        if non_virtual_cash > actual_cash + 1.0:
            self.logger.error(
                f'校验失败: 现金簿记合计(非虚拟)={non_virtual_cash:.2f} > 实际={actual_cash:.2f}'
            )
            raise ValueError(
                f'VirtualBook 校验失败: 现金簿记合计({non_virtual_cash:.2f}) '
                f'超过账户实际现金({actual_cash:.2f})'
            )
        if virtual_cash_total > 0:
            self.logger.info(
                f'虚拟资金实例: {[(i, f"{c:.2f}") for i, c in virtual_cash_instances]}, '
                f'合计={virtual_cash_total:.2f} (不参与账户现金校验)'
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

    def _register_all_instances(self):
        for config in self._configs:
            instance_id = config['instance_id']
            self._instance_meta[instance_id] = {
                'pid': os.getpid(),
                'start_time': time.time(),
                'start_time_str': time.strftime('%Y-%m-%d %H:%M:%S'),
                'strategy_name': config.get('strategy_name', ''),
                'status': 'running',
            }
            self._write_heartbeat(instance_id)
        self.logger.info(f'已注册 {len(self._instance_meta)} 个实例的心跳信息')

    def _write_heartbeat(self, instance_id: str):
        heartbeat_dir = os.path.join('logs', 'instances', instance_id)
        os.makedirs(heartbeat_dir, exist_ok=True)
        heartbeat_file = os.path.join(heartbeat_dir, 'heartbeat.json')
        meta = self._instance_meta.get(instance_id, {})
        data = {
            'instance_id': instance_id,
            'pid': meta.get('pid'),
            'start_time': meta.get('start_time_str', ''),
            'last_heartbeat': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': meta.get('status', 'unknown'),
        }
        try:
            with open(heartbeat_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.warning(f'写入心跳文件失败: {instance_id}, {e}')

    def _check_instance_alive(self, instance_id: str) -> bool:
        meta = self._instance_meta.get(instance_id)
        if not meta:
            return False
        pid = meta.get('pid')
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _restart_instance(self, instance_id: str):
        if self._restart_counts[instance_id] >= self._max_restart_attempts:
            self.logger.error(
                f'实例 {instance_id} 已达最大重试次数 '
                f'({self._max_restart_attempts})，停止重启'
            )
            if instance_id in self._instance_meta:
                self._instance_meta[instance_id]['status'] = 'failed'
            return

        self._restart_counts[instance_id] += 1
        self.logger.info(
            f'尝试重启实例 {instance_id} '
            f'(第{self._restart_counts[instance_id]}次)'
        )

        config = None
        for c in self._configs:
            if c['instance_id'] == instance_id:
                config = c
                break
        if not config:
            return

        try:
            strategy_name = config['strategy_name']
            strategy_class = get_strategy(strategy_name)
            kwargs = dict(get_strategy_default_kwargs(strategy_name))
            kwargs.update(config.get('kwargs', {}))

            account_id = config.get('account_id', 'default')
            mode = config.get('mode', 'sim')
            account_key = f"{account_id}_{mode}"
            api = self._apis.get(account_key)

            if api:
                book = self._books.get(instance_id)
                if book:
                    api.add_strategy(
                        strategy_class,
                        instance_id=instance_id,
                        virtual_book=book,
                        **kwargs
                    )
                    self._instance_meta[instance_id]['status'] = 'running'
                    self._instance_meta[instance_id]['start_time'] = time.time()
                    self._instance_meta[instance_id]['start_time_str'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    self._write_heartbeat(instance_id)
                    self.logger.info(f'实例 {instance_id} 重启成功')
        except Exception as e:
            self.logger.error(f'实例 {instance_id} 重启失败: {e}')
            if instance_id in self._instance_meta:
                self._instance_meta[instance_id]['status'] = 'failed'

    def _start_heartbeat(self):
        def _heartbeat_loop():
            while self._running:
                for instance_id in list(self._instance_meta.keys()):
                    meta = self._instance_meta[instance_id]
                    if meta.get('status') != 'running':
                        continue
                    self._write_heartbeat(instance_id)
                    if not self._check_instance_alive(instance_id):
                        self.logger.warning(f'实例 {instance_id} 心跳检测异常，尝试重启')
                        meta['status'] = 'dead'
                        self._restart_instance(instance_id)
                time.sleep(self._heartbeat_interval)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info(f'心跳检测线程已启动 (间隔{self._heartbeat_interval}秒)')

    def get_instance_health(self) -> List[dict]:
        result = []
        for instance_id, meta in self._instance_meta.items():
            heartbeat_dir = os.path.join('logs', 'instances', instance_id)
            heartbeat_file = os.path.join(heartbeat_dir, 'heartbeat.json')
            heartbeat_data = {}
            if os.path.isfile(heartbeat_file):
                try:
                    with open(heartbeat_file, 'r', encoding='utf-8') as f:
                        heartbeat_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
            result.append({
                'instance_id': instance_id,
                'strategy_name': meta.get('strategy_name', ''),
                'pid': meta.get('pid'),
                'start_time': meta.get('start_time_str', ''),
                'status': meta.get('status', 'unknown'),
                'restart_count': self._restart_counts.get(instance_id, 0),
                'last_heartbeat': heartbeat_data.get('last_heartbeat', ''),
            })
        return result

import os
import csv
import json
import logging
from typing import Optional, Dict
import datetime as dt_module


class RecordManager:
    """记录管理器 - 负责策略实例的状态持久化与交易记录

    职责：
    - 保存/加载策略状态（VirtualBook + StrategyLogic 组合状态）到 JSON
    - 追加成交记录到 CSV
    - 记录每日统计（当日覆盖）

    设计要点：
    - 使用原子写入（temp + os.replace）防止崩溃导致文件损坏
    - 交易过程文件统一保存到 trading_records/{instance_id}/ 目录
      （heartbeat 等运行时元数据仍在 logs/instances/{instance_id}/）
    - JSON 序列化使用 default=str 兜底不可序列化类型
    """

    # 成交记录 CSV 表头
    TRADES_HEADER = [
        '时间', '实例ID', '标的', '方向', '价格', '数量', '手续费', '订单ID'
    ]
    # 每日统计 CSV 表头
    DAILY_STATS_HEADER = [
        '日期', '现金', '持仓数量', '总市值'
    ]

    def __init__(self, instance_id: str, records_dir: str = 'trading_records'):
        """初始化记录管理器

        Args:
            instance_id: 策略实例ID
            records_dir: 记录文件根目录（默认 trading_records）
        """
        self.instance_id = instance_id
        self.records_dir = os.path.join(records_dir, instance_id)
        os.makedirs(self.records_dir, exist_ok=True)
        self.state_file = os.path.join(
            self.records_dir, f'state_{instance_id}.json'
        )
        self.trades_file = os.path.join(
            self.records_dir, f'trades_{instance_id}.csv'
        )
        self.daily_stats_file = os.path.join(
            self.records_dir, f'daily_stats_{instance_id}.csv'
        )
        self.logger = logging.getLogger(
            self.__class__.__module__ + '.' + self.__class__.__name__
        )

    def save_state(self, virtual_book_state: dict, strategy_state: dict):
        """保存策略状态到 JSON 文件（原子写入）

        Args:
            virtual_book_state: VirtualBook.get_state() 返回的字典
            strategy_state: StrategyLogic.get_state() 返回的字典
        """
        state_data = {
            'instance_id': self.instance_id,
            'saved_at': dt_module.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            'virtual_book': virtual_book_state,
            'strategy': strategy_state,
        }
        tmp_file = self.state_file + '.tmp'
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    state_data, f, indent=2,
                    ensure_ascii=False, default=str
                )
            os.replace(tmp_file, self.state_file)
            self.logger.debug(
                f'[{self.instance_id}] 状态已保存: '
                f'持仓={len(virtual_book_state.get("positions", {}))}只, '
                f'现金={virtual_book_state.get("cash", 0):.2f}'
            )
        except Exception as e:
            self.logger.error(f'[{self.instance_id}] 保存状态失败: {e}')
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

    def load_state(self) -> Optional[dict]:
        """从 JSON 文件加载策略状态

        Returns:
            包含 'virtual_book' 和 'strategy' 键的字典；文件不存在或解析失败返回 None
        """
        if not os.path.exists(self.state_file):
            self.logger.info(
                f'[{self.instance_id}] 状态文件不存在，将使用默认初始状态'
            )
            return None
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            self.logger.info(
                f'[{self.instance_id}] 已加载持久化状态 '
                f'(保存于 {state_data.get("saved_at", "未知")})'
            )
            return state_data
        except json.JSONDecodeError as e:
            self.logger.error(
                f'[{self.instance_id}] 解析状态文件失败: {e}，将使用默认初始状态'
            )
            return None
        except IOError as e:
            self.logger.error(
                f'[{self.instance_id}] 读取状态文件失败: {e}，将使用默认初始状态'
            )
            return None

    def append_trade(self, trade_record: dict):
        """追加成交记录到 CSV 文件

        Args:
            trade_record: 成交记录字典，包含键:
                - time: 时间字符串
                - instance_id: 实例ID
                - symbol: 标的代码
                - direction: 方向 (buy/sell)
                - price: 成交价格
                - volume: 成交数量
                - commission: 手续费
                - order_id: 订单ID
        """
        direction_cn = '买入' if trade_record.get('direction') == 'buy' else '卖出'
        row = [
            trade_record.get('time', ''),
            trade_record.get('instance_id', self.instance_id),
            trade_record.get('symbol', ''),
            direction_cn,
            trade_record.get('price', 0),
            trade_record.get('volume', 0),
            trade_record.get('commission', 0),
            trade_record.get('order_id', ''),
        ]
        write_header = not os.path.exists(self.trades_file)
        try:
            with open(
                self.trades_file, 'a', encoding='utf-8-sig', newline=''
            ) as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.TRADES_HEADER)
                writer.writerow(row)
        except IOError as e:
            self.logger.error(f'[{self.instance_id}] 追加成交记录失败: {e}')

    def save_daily_stats(
        self, cash: float, positions_count: int, total_value: float
    ):
        """记录每日统计数据（当日行覆盖）

        同一交易日多次保存会覆盖当日行，避免重复记录。

        Args:
            cash: 当前现金
            positions_count: 持仓数量
            total_value: 总市值
        """
        current_date = dt_module.datetime.now().strftime('%Y-%m-%d')
        data = {
            '日期': current_date,
            '现金': round(cash, 2),
            '持仓数量': positions_count,
            '总市值': round(total_value, 2),
        }
        try:
            if not os.path.exists(self.daily_stats_file):
                # 首次创建
                with open(
                    self.daily_stats_file, 'w',
                    encoding='utf-8-sig', newline=''
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=self.DAILY_STATS_HEADER)
                    writer.writeheader()
                    writer.writerow(data)
            else:
                # 读取已有数据，移除当日行，追加新行
                existing_rows = []
                with open(
                    self.daily_stats_file, 'r', encoding='utf-8-sig', newline=''
                ) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('日期') != current_date:
                            existing_rows.append(row)
                existing_rows.append(data)
                with open(
                    self.daily_stats_file, 'w',
                    encoding='utf-8-sig', newline=''
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=self.DAILY_STATS_HEADER)
                    writer.writeheader()
                    writer.writerows(existing_rows)
            self.logger.debug(
                f'[{self.instance_id}] 每日统计已记录: '
                f'日期={current_date}, 现金={cash:.2f}, '
                f'持仓={positions_count}只, 总市值={total_value:.2f}'
            )
        except Exception as e:
            self.logger.error(f'[{self.instance_id}] 记录每日统计失败: {e}')

from typing import Dict, Type, Any, Optional
import importlib
import pkgutil
import os

_STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_strategy(name: str, default_kwargs: Optional[Dict[str, Any]] = None,
                      backtest_config: Optional[Dict[str, Any]] = None):
    """策略注册装饰器

    Args:
        name: 策略注册名称，用于命令行选择策略
        default_kwargs: 策略默认参数，如 {'fast_period': 5, 'slow_period': 20}
        backtest_config: 回测推荐配置，如 {'cash': 200000, 'commission': 0.0001,
                         'start_date': '2025-07-10', 'end_date': '2026-04-17'}
    """
    def decorator(cls):
        _STRATEGY_REGISTRY[name] = {
            'class': cls,
            'default_kwargs': default_kwargs or {},
            'backtest_config': backtest_config or {},
        }
        return cls
    return decorator


def get_strategy(name: str) -> Optional[Type]:
    """获取策略类"""
    _auto_discover()
    entry = _STRATEGY_REGISTRY.get(name)
    return entry['class'] if entry else None


def get_strategy_default_kwargs(name: str) -> Dict[str, Any]:
    """获取策略默认参数"""
    _auto_discover()
    entry = _STRATEGY_REGISTRY.get(name)
    return entry.get('default_kwargs', {}) if entry else {}


def get_strategy_backtest_config(name: str) -> Dict[str, Any]:
    """获取策略回测推荐配置"""
    _auto_discover()
    entry = _STRATEGY_REGISTRY.get(name)
    return entry.get('backtest_config', {}) if entry else {}


def get_all_strategy_names() -> list:
    """获取所有已注册的策略名称"""
    _auto_discover()
    return list(_STRATEGY_REGISTRY.keys())


def get_strategy_choices() -> list:
    """获取策略选项列表，用于argparse的choices参数"""
    return get_all_strategy_names()


_DISCOVERED = False


def _auto_discover():
    """自动发现 strategies 包下的所有策略模块"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    strategies_dir = os.path.dirname(__file__)
    for _, module_name, is_pkg in pkgutil.iter_modules([strategies_dir]):
        if module_name.startswith('_') or module_name == 'config':
            continue
        try:
            importlib.import_module(f'strategies.{module_name}')
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"自动发现策略模块失败: strategies.{module_name}, {e}")

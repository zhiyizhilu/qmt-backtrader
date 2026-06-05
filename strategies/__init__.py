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
                         'slippage': 0.001, 'start_date': '2025-07-10',
                         'end_date': '2026-04-17'}
    """
    def decorator(cls):
        import inspect
        module = inspect.getmodule(cls)
        strategy_dir = os.path.dirname(os.path.abspath(module.__file__)) if module else None
        _STRATEGY_REGISTRY[name] = {
            'class': cls,
            'default_kwargs': default_kwargs or {},
            'backtest_config': backtest_config or {},
            'strategy_dir': strategy_dir,
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


def get_strategy_dir(name: str) -> Optional[str]:
    """获取策略所在目录的绝对路径"""
    _auto_discover()
    entry = _STRATEGY_REGISTRY.get(name)
    return entry.get('strategy_dir') if entry else None


_DISCOVERED = False


def _auto_discover():
    """自动发现 strategies、strategies_for_vip、strategies_for_svip 及 strategies_my 包下的所有策略模块"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    _discover_package('strategies', os.path.dirname(__file__))
    vip_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies_for_vip')
    vip_dir = os.path.normpath(vip_dir)
    if os.path.isdir(vip_dir):
        _discover_package('strategies_for_vip', vip_dir)
    svip_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies_for_svip')
    svip_dir = os.path.normpath(svip_dir)
    if os.path.isdir(svip_dir):
        _discover_package('strategies_for_svip', svip_dir)
    my_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies_my')
    my_dir = os.path.normpath(my_dir)
    if os.path.isdir(my_dir):
        _discover_package('strategies_my', my_dir)


def _discover_package(package_name: str, package_dir: str):
    """发现指定包下的所有策略模块"""
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if module_name.startswith('_') or module_name == 'config':
            continue
        try:
            importlib.import_module(f'{package_name}.{module_name}')
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"自动发现策略模块失败: {package_name}.{module_name}, {e}")

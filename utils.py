"""
工具函数模块
包含日志、内存管理、时间监控等通用工具函数
"""

import os
import gc
import time
import psutil
from contextlib import contextmanager
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def log(log_str: str) -> None:
    """打印日志"""
    from config import ENABLE_LOG
    if ENABLE_LOG:
        print(log_str)


def g() -> int:
    """执行垃圾回收"""
    return gc.collect()


def delete(*obj_list) -> None:
    """删除对象并执行垃圾回收"""
    for obj in obj_list:
        del obj
    gc_cnt = g()
    if gc_cnt > 0:
        log(f"unreachable_obj_found: {gc_cnt}")


@contextmanager
def timer_memory(name: str):
    """计时器和内存监控上下文管理器"""
    t0 = time.time()
    yield
    print(f'Memory: {(psutil.Process(os.getpid()).memory_info().rss / 2**30):.02f}GB')
    print(f'{name} done in {time.time() - t0:.0f}s')


def robust_boxcox(data: pd.Series, lmbda: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    稳健的Box-Cox变换
    当lmbda接近0时使用log1p变换避免数值问题
    
    Args:
        data: 输入数据
        lmbda: Box-Cox变换参数，如果为None则自动估计
    
    Returns:
        (变换后的数据, lambda参数)
    """
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    
    if lmbda is None:
        transformed, lmbda = boxcox(1 + data)
        if lmbda <= 0:
            lmbda = 0
            transformed = np.log1p(data)
        return transformed, lmbda
    else:
        if lmbda <= 0:
            lmbda = 0
            transformed = np.log1p(data)
        else:
            transformed = boxcox(1 + data, lmbda)
        return transformed, lmbda


def robust_inv_boxcox(data: pd.Series, lmbda: float) -> np.ndarray:
    """
    稳健的Box-Cox逆变换
    
    Args:
        data: 变换后的数据
        lmbda: Box-Cox变换参数
    
    Returns:
        原始数据
    """
    from scipy.special import inv_boxcox
    
    if lmbda <= 0:
        return np.expm1(data)
    else:
        return inv_boxcox(data, lmbda) - 1

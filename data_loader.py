"""
数据加载与预处理模块
包含数据读取、降采样等功能
"""

from typing import Tuple

import pandas as pd

from utils import log, delete


def read_data_file(file_path: str, is_test_file: bool = False) -> pd.DataFrame:
    """
    读取数据文件
    
    Args:
        file_path: 文件路径
        is_test_file: 是否为测试文件
    
    Returns:
        DataFrame
    """
    log(f'read file [is_test_file={is_test_file}]: {file_path}')
    
    if is_test_file:
        df = pd.read_csv(file_path, parse_dates=['click_time'])
    else:
        df = pd.read_csv(
            file_path,
            parse_dates=['click_time', 'attributed_time'],
            dtype={'is_attributed': 'bool'}
        )
    return df


def random_down_sample(
    df: pd.DataFrame,
    majority_multiply: int = 1,
    target_col_name: str = 'is_attributed',
    minority_val: int = 1,
    majority_val: int = 0
) -> pd.DataFrame:
    """
    随机降采样：使多数类和少数类样本比例达到指定比例
    
    Args:
        df: 输入数据框
        majority_multiply: 多数类相对于少数类的倍数
        target_col_name: 目标列名
        minority_val: 少数类标签值
        majority_val: 多数类标签值
    
    Returns:
        降采样后的数据框
    """
    df_minority = df[df[target_col_name] == minority_val]
    df_majority = df[df[target_col_name] == majority_val]
    
    minority_count = len(df_minority)
    majority_count = len(df_majority)
    
    majority_sample_size = minority_count * majority_multiply
    
    if majority_sample_size >= majority_count:
        log(f'warning: majority_sample_size({majority_sample_size}) >= majority_count({majority_count})')
        return df
    
    df_majority_sampled = df_majority.sample(n=majority_sample_size, random_state=42)
    
    df_downsampled = pd.concat([df_minority, df_majority_sampled], ignore_index=True)
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    delete(df_minority, df_majority, df_majority_sampled)
    
    return df_downsampled


def feature_target_split(
    df: pd.DataFrame,
    target_col_name: str = 'is_attributed',
    inplace: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    分割特征和目标变量
    
    Args:
        df: 输入数据框
        target_col_name: 目标列名
        inplace: 是否原地修改
    
    Returns:
        (特征数据框, 目标序列)
    """
    from utils import g
    
    y = df[target_col_name]
    g()
    
    if inplace:
        df.drop(target_col_name, axis=1, inplace=True)
        g()
        X = df
    else:
        X = df.drop(target_col_name, axis=1)
        g()
    
    return X, y

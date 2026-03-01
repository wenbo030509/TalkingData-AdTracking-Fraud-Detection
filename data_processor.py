"""
数据处理模块
包含IP桶处理、数据集合并等功能
"""

from typing import Tuple

import pandas as pd

from config import (
    TRAIN_BUCKET_DIR, TEST_BUCKET_DIR, IP_BUCKET_NUM,
    VLDT_SET_SIZE, IS_DOWN_SAMPLE, MAJORITY_MULTIPLY,
    NON_TRAIN_COLUMNS
)
from data_loader import read_data_file, random_down_sample
from feature_engineering import extract_datetime_features, add_features
from utils import log, delete, g


def process_ip_bucket(
    ip_bucket: int,
    is_down_sample: bool = True,
    majority_multiply: int = 1,
    tsfm_sv_policy: str = 'new_and_save'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    处理单个IP桶的数据
    
    Args:
        ip_bucket: IP桶编号（0-19）
        is_down_sample: 是否对训练集降采样
        majority_multiply: 多数类采样倍数
        tsfm_sv_policy: 变换器保存策略
    
    Returns:
        (训练集, 验证集, 测试集)
    """
    log(f'---------- process bucket: {ip_bucket} -----------')
    
    train_file_path = f'{TRAIN_BUCKET_DIR}/train_{ip_bucket}.csv'
    test_file_path = f'{TEST_BUCKET_DIR}/test_{ip_bucket}.csv'
    
    df_test = read_data_file(test_file_path, is_test_file=True)
    df_train_vldt = read_data_file(train_file_path, is_test_file=False)
    
    log(f'read files: {train_file_path}; {test_file_path}')
    
    df_train_vldt['click_id'] = -1
    df_test['is_attributed'] = -1
    df_full = pd.concat([df_train_vldt, df_test], sort=False)
    
    train_vldt_len = len(df_train_vldt)
    delete(df_train_vldt, df_test)
    
    log('add features: ')
    df_full = extract_datetime_features(df_full)
    df_full = add_features(df_full, save_transformer=tsfm_sv_policy)
    g()
    
    log(f'drop non-training-columns: {NON_TRAIN_COLUMNS}')
    df_full.drop(NON_TRAIN_COLUMNS, axis=1, inplace=True)
    g()
    
    log('split data set: ')
    vldt_len = min(VLDT_SET_SIZE // IP_BUCKET_NUM, train_vldt_len // 5)
    
    df_train = df_full[:train_vldt_len - vldt_len].copy()
    df_vldt = df_full[train_vldt_len - vldt_len:train_vldt_len].copy()
    df_test = df_full[train_vldt_len:].copy()
    
    log('drop temporary columns: ')
    df_train.drop(['click_id'], axis=1, inplace=True)
    df_vldt.drop(['click_id'], axis=1, inplace=True)
    df_test.drop(['is_attributed'], axis=1, inplace=True)
    
    log(f'shape: vldt={df_vldt.shape}; test={df_test.shape}')
    log(f'shape: train(before downsample)={df_train.shape}')
    g()
    
    if is_down_sample:
        log(f'down_sample: majority_multiply={majority_multiply}')
        df_train = random_down_sample(df_train, majority_multiply)
        log('gc.collect (warning is OK): ')
        g()
        log(f'shape(after downsample)={df_train.shape}')
    
    return df_train, df_vldt, df_test


def prep_data_set_full_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    准备完整数据集（合并所有IP桶）
    
    Returns:
        (训练集, 验证集, 测试集)
    """
    log_template = "append bkt {}: train.shape={}; vldt.shape={}; test.shape={}"
    
    df_train, df_vldt, df_test = process_ip_bucket(
        ip_bucket=0,
        is_down_sample=IS_DOWN_SAMPLE,
        majority_multiply=MAJORITY_MULTIPLY,
        tsfm_sv_policy='new_and_save'
    )
    log(log_template.format(0, df_train.shape, df_vldt.shape, df_test.shape))
    
    for bkt_id in range(1, IP_BUCKET_NUM):
        train_bkt, vldt_bkt, test_bkt = process_ip_bucket(
            ip_bucket=bkt_id,
            is_down_sample=IS_DOWN_SAMPLE,
            majority_multiply=MAJORITY_MULTIPLY,
            tsfm_sv_policy='reuse'
        )
        g()
        
        df_train = pd.concat([df_train, train_bkt], ignore_index=True)
        g()
        df_vldt = pd.concat([df_vldt, vldt_bkt], ignore_index=True)
        g()
        df_test = pd.concat([df_test, test_bkt], ignore_index=True)
        g()
        
        delete(train_bkt, vldt_bkt, test_bkt)
        log(log_template.format(bkt_id, df_train.shape, df_vldt.shape, df_test.shape))
    
    df_test.set_index('click_id', drop=True, inplace=True)
    df_test.sort_index(axis=0, inplace=True)
    
    return df_train, df_vldt, df_test


def prep_feature_target_full_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    准备完整的特征和目标变量
    
    Returns:
        (X_train, y_train, X_vldt, y_vldt, df_test)
    """
    from config import CATEGORICAL_FEATURES
    from data_loader import feature_target_split
    
    df_train, df_vldt, df_test = prep_data_set_full_data()
    
    X_train, y_train = feature_target_split(df_train, target_col_name='is_attributed', inplace=True)
    X_vldt, y_vldt = feature_target_split(df_vldt, target_col_name='is_attributed', inplace=True)
    
    delete(df_train, df_vldt)
    
    log('----: prep_feature_target_full_data :-----')
    log(f'-- X_train={X_train.shape}; y_train={y_train.shape}; X_vldt={X_vldt.shape}; y_vldt={y_vldt.shape}; df_test={df_test.shape}')
    log(f'-- features: {X_train.columns.values.tolist()}')
    log(f'-- categorical: {CATEGORICAL_FEATURES}')
    log(f'-- y_train.value_counts:\n{y_train.value_counts()}')
    log(f'-- y_vldt.value_counts:\n{y_vldt.value_counts()}')
    
    return X_train, y_train, X_vldt, y_vldt, df_test

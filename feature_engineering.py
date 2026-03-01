"""
特征工程模块
包含时间特征、统计特征、去重特征、时间间隔特征、累积计数特征的提取
"""

import pandas as pd

from utils import log


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取时间相关特征
    
    Args:
        df: 输入数据框
    
    Returns:
        添加时间特征后的数据框
    """
    df['dd'] = df['click_time'].dt.day
    df['hh'] = df['click_time'].dt.hour
    return df


def extract_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取计数统计特征
    
    Args:
        df: 输入数据框
    
    Returns:
        添加计数特征后的数据框
    """
    df['cnt_grp_by_ip_device'] = df.groupby(['ip', 'device'])['channel'].transform('count')
    df['cnt_grp_by_ip_app'] = df.groupby(['ip', 'app'])['channel'].transform('count')
    df['cnt_grp_by_ip_hh_app'] = df.groupby(['ip', 'hh', 'app'])['channel'].transform('count')
    df['cnt_grp_by_ip_hh_device'] = df.groupby(['ip', 'hh', 'device'])['channel'].transform('count')
    df['cnt_grp_by_app_channel'] = df.groupby(['app', 'channel'])['ip'].transform('count')
    df['cnt_grp_by_dd_hh_app_channel'] = df.groupby(['dd', 'hh', 'app', 'channel'])['ip'].transform('count')
    
    return df


def extract_nunique_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取去重计数特征
    
    Args:
        df: 输入数据框
    
    Returns:
        添加去重特征后的数据框
    """
    df['nunique_on_channel_grp_by_ip'] = df.groupby('ip')['channel'].transform('nunique')
    df['nunique_on_app_grp_by_ip'] = df.groupby('ip')['app'].transform('nunique')
    df['nunique_on_app_grp_by_ip_hh'] = df.groupby(['ip', 'hh'])['app'].transform('nunique')
    df['nunique_on_channel_grp_by_app'] = df.groupby('app')['channel'].transform('nunique')
    df['nunique_on_channel_grp_by_hh_app'] = df.groupby(['hh', 'app'])['channel'].transform('nunique')
    
    return df


def extract_next_interval_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取下次点击间隔特征
    
    Args:
        df: 输入数据框
    
    Returns:
        添加时间间隔特征后的数据框
    """
    df['nxt_itvl_by_ip_channel'] = df.groupby(['ip', 'channel'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_channel'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_channel'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_channel: max={df["nxt_itvl_by_ip_channel"].max()}; min={df["nxt_itvl_by_ip_channel"].min()}; mean={df["nxt_itvl_by_ip_channel"].mean()}')
    
    df['nxt_itvl_by_ip_app_channel'] = df.groupby(['ip', 'app', 'channel'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_app_channel'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_app_channel'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_app_channel: max={df["nxt_itvl_by_ip_app_channel"].max()}; min={df["nxt_itvl_by_ip_app_channel"].min()}; mean={df["nxt_itvl_by_ip_app_channel"].mean()}')
    
    df['nxt_itvl_by_ip_os_device_app'] = df.groupby(['ip', 'os', 'device', 'app'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_os_device_app'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_os_device_app'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_os_device_app: max={df["nxt_itvl_by_ip_os_device_app"].max()}; min={df["nxt_itvl_by_ip_os_device_app"].min()}; mean={df["nxt_itvl_by_ip_os_device_app"].mean()}')
    
    return df


def extract_cumcount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取累积计数特征
    
    Args:
        df: 输入数据框
    
    Returns:
        添加累积计数特征后的数据框
    """
    df['cumcount_on_app_grp_by_ip_device_os'] = df.groupby(['ip', 'device', 'os']).cumcount()
    log(f'\tcumcount_on_app_grp_by_ip_device_os: max={df["cumcount_on_app_grp_by_ip_device_os"].max()}; min={df["cumcount_on_app_grp_by_ip_device_os"].min()}; mean={df["cumcount_on_app_grp_by_ip_device_os"].mean()}')
    
    return df


def add_features(df: pd.DataFrame, save_transformer: str = 'new_and_save') -> pd.DataFrame:
    """
    添加所有特征
    
    Args:
        df: 输入数据框
        save_transformer: 变换器保存策略（'new_and_save'或'reuse'）
    
    Returns:
        添加所有特征后的数据框
    """
    log('add feature:  nxt_itvl_by_ip_os_device_app')
    df = extract_next_interval_features(df)
    
    log('add feature:  cnt_grp_by_dd_hh_app_channel')
    df = extract_count_features(df)
    
    log('add feature:  cumcount_on_app_grp_by_ip_device_os')
    df = extract_cumcount_features(df)
    
    log('add feature:  nunique features')
    df = extract_nunique_features(df)
    
    return df

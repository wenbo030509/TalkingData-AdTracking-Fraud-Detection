"""
TalkingData AdTracking Fraud Detection
======================================
基于LightGBM的广告欺诈检测模型

任务目标：预测用户点击广告后是否会下载安装APP（is_attributed=True）
核心挑战：
1. 极端不平衡数据集（正例/负例 ≈ 0.03/1）
2. 内存限制（数据集过大）
3. 验证集需要保持原始分布以反映真实效果

"""

import os
import gc
import time
import warnings
from contextlib import contextmanager
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import psutil
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# =============================================================================
# 全局配置
# =============================================================================

# 数据路径配置
TRAIN_BUCKET_DIR = '/kaggle/input/train'
TEST_BUCKET_DIR = '/kaggle/input/test'
OUTPUT_DIR = '/kaggle/output'

# 数据处理配置
g_ip_bkt_num = 20                    # IP分桶数量
g_is_down_sample = True              # 是否对训练集降采样
g_majority_multiply = 1              # 多数类采样倍数（正例:负例 = 1:majority_multiply）
g_vldt_set_size = 5000000            # 验证集总大小
g_scale_pos_weight = None            # LightGBM的类别权重（None表示自动计算）

# 类别特征列表
g_categorical_features = ['app', 'device', 'os', 'channel', 'hh']

# 非训练列（需要删除的列）
g_non_train_columns = ['click_time', 'dd', 'ip']

# 日志开关
g_enable_log = True


# =============================================================================
# 工具函数
# =============================================================================

def log(log_str: str) -> None:
    """打印日志"""
    if g_enable_log:
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
    """
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    
    if lmbda is None:
        transformed, lmbda = boxcox(1 + data)
        # 处理boxcox返回接近0的负数的情况
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
    """稳健的Box-Cox逆变换"""
    from scipy.special import inv_boxcox
    
    if lmbda <= 0:
        return np.expm1(data)
    else:
        return inv_boxcox(data, lmbda) - 1


# =============================================================================
# 数据加载与预处理
# =============================================================================

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
    
    # 计算需要采样的多数类样本数
    majority_sample_size = minority_count * majority_multiply
    
    if majority_sample_size >= majority_count:
        log(f'warning: majority_sample_size({majority_sample_size}) >= majority_count({majority_count})')
        return df
    
    # 随机采样多数类
    df_majority_sampled = df_majority.sample(n=majority_sample_size, random_state=42)
    
    # 合并并打乱顺序
    df_downsampled = pd.concat([df_minority, df_majority_sampled], ignore_index=True)
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    delete(df_minority, df_majority, df_majority_sampled)
    
    return df_downsampled


# =============================================================================
# 特征工程
# =============================================================================

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取时间相关特征"""
    df['dd'] = df['click_time'].dt.day
    df['hh'] = df['click_time'].dt.hour
    return df


def extract_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取计数统计特征"""
    # 按IP-Device分组计数
    df['cnt_grp_by_ip_device'] = df.groupby(['ip', 'device'])['channel'].transform('count')
    
    # 按IP-App分组计数
    df['cnt_grp_by_ip_app'] = df.groupby(['ip', 'app'])['channel'].transform('count')
    
    # 按IP-小时-App分组计数
    df['cnt_grp_by_ip_hh_app'] = df.groupby(['ip', 'hh', 'app'])['channel'].transform('count')
    
    # 按IP-小时-Device分组计数
    df['cnt_grp_by_ip_hh_device'] = df.groupby(['ip', 'hh', 'device'])['channel'].transform('count')
    
    # 按App-Channel分组计数
    df['cnt_grp_by_app_channel'] = df.groupby(['app', 'channel'])['ip'].transform('count')
    
    # 按日期-小时-App-Channel分组计数
    df['cnt_grp_by_dd_hh_app_channel'] = df.groupby(['dd', 'hh', 'app', 'channel'])['ip'].transform('count')
    
    return df


def extract_nunique_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取去重计数特征"""
    # 按IP分组，Channel去重数
    df['nunique_on_channel_grp_by_ip'] = df.groupby('ip')['channel'].transform('nunique')
    
    # 按IP分组，App去重数
    df['nunique_on_app_grp_by_ip'] = df.groupby('ip')['app'].transform('nunique')
    
    # 按IP-小时分组，App去重数
    df['nunique_on_app_grp_by_ip_hh'] = df.groupby(['ip', 'hh'])['app'].transform('nunique')
    
    # 按App分组，Channel去重数
    df['nunique_on_channel_grp_by_app'] = df.groupby('app')['channel'].transform('nunique')
    
    # 按小时-App分组，Channel去重数
    df['nunique_on_channel_grp_by_hh_app'] = df.groupby(['hh', 'app'])['channel'].transform('nunique')
    
    return df


def extract_next_interval_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取下次点击间隔特征"""
    # 按IP-Channel分组，计算下次点击时间间隔
    df['nxt_itvl_by_ip_channel'] = df.groupby(['ip', 'channel'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_channel'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_channel'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_channel: max={df["nxt_itvl_by_ip_channel"].max()}; min={df["nxt_itvl_by_ip_channel"].min()}; mean={df["nxt_itvl_by_ip_channel"].mean()}')
    
    # 按IP-App-Channel分组，计算下次点击时间间隔
    df['nxt_itvl_by_ip_app_channel'] = df.groupby(['ip', 'app', 'channel'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_app_channel'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_app_channel'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_app_channel: max={df["nxt_itvl_by_ip_app_channel"].max()}; min={df["nxt_itvl_by_ip_app_channel"].min()}; mean={df["nxt_itvl_by_ip_app_channel"].mean()}')
    
    # 按IP-OS-Device-App分组，计算下次点击时间间隔
    df['nxt_itvl_by_ip_os_device_app'] = df.groupby(['ip', 'os', 'device', 'app'])['click_time'].diff().shift(-1).dt.seconds
    fill_value = df['nxt_itvl_by_ip_os_device_app'].mean()
    log(f'\tfillna: {fill_value}')
    df['nxt_itvl_by_ip_os_device_app'].fillna(fill_value, inplace=True)
    log(f'\tnxt_itvl_by_ip_os_device_app: max={df["nxt_itvl_by_ip_os_device_app"].max()}; min={df["nxt_itvl_by_ip_os_device_app"].min()}; mean={df["nxt_itvl_by_ip_os_device_app"].mean()}')
    
    return df


def extract_cumcount_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取累积计数特征"""
    # 按IP-Device-OS分组，App的累积计数
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
        添加特征后的数据框
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


# =============================================================================
# 数据分桶处理
# =============================================================================

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
    
    # 读取文件
    train_file_path = f'{TRAIN_BUCKET_DIR}/train_{ip_bucket}.csv'
    test_file_path = f'{TEST_BUCKET_DIR}/test_{ip_bucket}.csv'
    
    df_test = read_data_file(test_file_path, is_test_file=True)
    df_train_vldt = read_data_file(train_file_path, is_test_file=False)
    
    log(f'read files: {train_file_path}; {test_file_path}')
    
    # 对齐列并合并
    df_train_vldt['click_id'] = -1
    df_test['is_attributed'] = -1
    df_full = pd.concat([df_train_vldt, df_test], sort=False)
    
    train_vldt_len = len(df_train_vldt)
    delete(df_train_vldt, df_test)
    
    # 特征工程
    log('add features: ')
    df_full = extract_datetime_features(df_full)
    df_full = add_features(df_full, save_transformer=tsfm_sv_policy)
    g()
    
    # 删除非训练列
    log(f'drop non-training-columns: {g_non_train_columns}')
    df_full.drop(g_non_train_columns, axis=1, inplace=True)
    g()
    
    # 分割数据集
    log('split data set: ')
    vldt_len = min(g_vldt_set_size // g_ip_bkt_num, train_vldt_len // 5)
    
    df_train = df_full[:train_vldt_len - vldt_len].copy()
    df_vldt = df_full[train_vldt_len - vldt_len:train_vldt_len].copy()
    df_test = df_full[train_vldt_len:].copy()
    
    # 删除临时列
    log('drop temporary columns: ')
    df_train.drop(['click_id'], axis=1, inplace=True)
    df_vldt.drop(['click_id'], axis=1, inplace=True)
    df_test.drop(['is_attributed'], axis=1, inplace=True)
    
    log(f'shape: vldt={df_vldt.shape}; test={df_test.shape}')
    log(f'shape: train(before downsample)={df_train.shape}')
    g()
    
    # 降采样
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
    
    # 处理第一个桶
    df_train, df_vldt, df_test = process_ip_bucket(
        ip_bucket=0,
        is_down_sample=g_is_down_sample,
        majority_multiply=g_majority_multiply,
        tsfm_sv_policy='new_and_save'
    )
    log(log_template.format(0, df_train.shape, df_vldt.shape, df_test.shape))
    
    # 处理剩余桶
    for bkt_id in range(1, g_ip_bkt_num):
        train_bkt, vldt_bkt, test_bkt = process_ip_bucket(
            ip_bucket=bkt_id,
            is_down_sample=g_is_down_sample,
            majority_multiply=g_majority_multiply,
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
    
    # 设置测试集索引
    df_test.set_index('click_id', drop=True, inplace=True)
    df_test.sort_index(axis=0, inplace=True)
    
    return df_train, df_vldt, df_test


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


def prep_feature_target_full_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    准备完整的特征和目标变量
    
    Returns:
        (X_train, y_train, X_vldt, y_vldt, df_test)
    """
    df_train, df_vldt, df_test = prep_data_set_full_data()
    
    X_train, y_train = feature_target_split(df_train, target_col_name='is_attributed', inplace=True)
    X_vldt, y_vldt = feature_target_split(df_vldt, target_col_name='is_attributed', inplace=True)
    
    delete(df_train, df_vldt)
    
    log('----: prep_feature_target_full_data :-----')
    log(f'-- X_train={X_train.shape}; y_train={y_train.shape}; X_vldt={X_vldt.shape}; y_vldt={y_vldt.shape}; df_test={df_test.shape}')
    log(f'-- features: {X_train.columns.values.tolist()}')
    log(f'-- categorical: {g_categorical_features}')
    log(f'-- y_train.value_counts:\n{y_train.value_counts()}')
    log(f'-- y_vldt.value_counts:\n{y_vldt.value_counts()}')
    
    return X_train, y_train, X_vldt, y_vldt, df_test


# =============================================================================
# 模型定义与训练
# =============================================================================

def default_model() -> LGBMClassifier:
    """创建默认的LightGBM模型"""
    lgb_default = LGBMClassifier()
    return lgb_default.set_params(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        verbose=1,
        nthread=4,
        iid=False,
        two_round=True
    )


def gbdt_base_001() -> LGBMClassifier:
    """基线模型配置"""
    return default_model().set_params(
        subsample=0.8,
        subsample_freq=1,
        subsample_for_bin=200000,
        colsample_bytree=0.8,
        learning_rate=0.08,
        num_leaves=105,
        max_depth=7,
        min_split_gain=0.3,
        max_bin=255,
        reg_alpha=0.3,
        n_estimators=2500
    )


# 基线模型字典
g_base_models = {
    'gbdt_base_001': gbdt_base_001()
}

# 网格搜索参数字典
g_search_params = {
    'gbdt_base_001_exp_002': {'min_sum_hessian_in_leaf': [0.001, 0.01, 0.05, 0.1]},
    'gbdt_base_001_exp_005': {'min_split_gain': [0.3, 0.4, 0.5]},
    'gbdt_base_001_exp_010': {'num_leaves': [90, 105], 'min_split_gain': [0.3, 0.4]}
}


def update_data_balancing_param(
    lgb_model: LGBMClassifier,
    y_value_counts: pd.Series,
    majority_val: int = 0,
    minority_val: int = 1
) -> LGBMClassifier:
    """
    更新数据平衡参数
    
    Args:
        lgb_model: LightGBM模型
        y_value_counts: 目标变量值计数
        majority_val: 多数类标签值
        minority_val: 少数类标签值
    
    Returns:
        更新后的模型
    """
    log(f'y_value_counts: \n{y_value_counts}')
    
    if g_scale_pos_weight is not None:
        log(f'use global config: scale_pos_weight={g_scale_pos_weight}')
        lgb_model.set_params(scale_pos_weight=g_scale_pos_weight)
    else:
        unbalance_degree = y_value_counts[majority_val] / y_value_counts[minority_val]
        log(f'majority_count/minority_count={unbalance_degree}')
        if unbalance_degree > 1.5 or unbalance_degree < 0.66:
            log('set is_unbalance=True')
            lgb_model.set_params(is_unbalance=True)
    
    return lgb_model


def get_model_and_search_params(base_model_id: str, search_params_id: str) -> Tuple[LGBMClassifier, Dict]:
    """获取模型和搜索参数"""
    return g_base_models[base_model_id], g_search_params[search_params_id]


# 模型训练参数
g_fit_params = {
    'categorical_feature': g_categorical_features,
    'early_stopping_rounds': 25,
    'verbose': 10,
    'eval_metric': 'auc'
}


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    lgb_model: LGBMClassifier
) -> LGBMClassifier:
    """
    训练LightGBM模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_validate: 验证特征
        y_validate: 验证目标
        lgb_model: LightGBM模型
    
    Returns:
        训练好的模型
    """
    # 获取类别特征索引
    cat_fea_indices = [X_train.columns.get_loc(col) for col in g_fit_params['categorical_feature']]
    
    log('------ fit parameters: ---------------------')
    log(f"early_stopping_rounds: {g_fit_params['early_stopping_rounds']}")
    log(f"verbose: {g_fit_params['verbose']}")
    log(f"eval_metric: {g_fit_params['eval_metric']}")
    log(f'categorical_feature: {cat_fea_indices}')
    for i in cat_fea_indices:
        log(f"\t{i}: {g_fit_params['categorical_feature'][i]}")
    
    log('------ update balancing parameters: ---------------------')
    lgb_model = update_data_balancing_param(lgb_model, y_train.value_counts())
    
    log('------ model parameters: ---------------------')
    log(lgb_model.get_params())
    
    # 训练模型
    fitted = lgb_model.fit(
        X_train, y_train,
        categorical_feature=cat_fea_indices,
        early_stopping_rounds=g_fit_params['early_stopping_rounds'],
        verbose=g_fit_params['verbose'],
        eval_metric=g_fit_params['eval_metric'],
        eval_set=[(X_validate, y_validate)]
    )
    
    g()
    return fitted


# =============================================================================
# 模型评估
# =============================================================================

def evaluate_model(
    model: LGBMClassifier,
    X_vldt: pd.DataFrame,
    y_vldt: pd.Series,
    threshold: float = 0.5
) -> Dict:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_vldt: 验证特征
        y_vldt: 验证目标
        threshold: 分类阈值
    
    Returns:
        评估指标字典
    """
    # 预测概率
    y_pred_proba = model.predict_proba(X_vldt)[:, 1]
    
    # 预测标签
    y_pred_label = (y_pred_proba >= threshold).astype(int)
    
    # 计算AUC
    auc = roc_auc_score(y_vldt, y_pred_proba)
    
    log(f'AUC: {auc:.6f}')
    log('\nClassification Report:')
    log(classification_report(y_vldt, y_pred_label, target_names=['not_attributed', 'is_attributed']))
    
    return {
        'auc': auc,
        'y_pred_proba': y_pred_proba,
        'y_pred_label': y_pred_label
    }


def plot_feature_importance(model: LGBMClassifier, importance_type: str = 'split'):
    """绘制特征重要性"""
    import matplotlib.pyplot as plt
    
    lgb.plot_importance(model.booster_, importance_type=importance_type, figsize=(10, 8))
    plt.title(f'Feature Importance ({importance_type})')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance_{importance_type}.png')
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：完整的训练和预测流程"""
    
    # 1. 准备数据
    log("=" * 60)
    log("Step 1: Preparing data...")
    log("=" * 60)
    
    with timer_memory('Data preparation'):
        X_train, y_train, X_vldt, y_vldt, df_test = prep_feature_target_full_data()
    
    # 2. 训练模型
    log("\n" + "=" * 60)
    log("Step 2: Training model...")
    log("=" * 60)
    
    # 获取基线模型
    base_model = g_base_models['gbdt_base_001']
    
    with timer_memory('Model training'):
        fitted_model = fit_model(X_train, y_train, X_vldt, y_vldt, base_model)
    
    # 3. 评估模型
    log("\n" + "=" * 60)
    log("Step 3: Evaluating model...")
    log("=" * 60)
    
    eval_results = evaluate_model(fitted_model, X_vldt, y_vldt)
    
    # 4. 生成预测结果
    log("\n" + "=" * 60)
    log("Step 4: Generating predictions...")
    log("=" * 60)
    
    y_test_pred_proba = fitted_model.predict_proba(df_test)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'click_id': df_test.index,
        'is_attributed': y_test_pred_proba
    })
    
    submission_path = f'{OUTPUT_DIR}/submission.csv'
    submission.to_csv(submission_path, index=False)
    log(f'Submission saved to: {submission_path}')
    log(f'Submission shape: {submission.shape}')
    log(f'Submission head:\n{submission.head()}')
    
    # 5. 特征重要性
    log("\n" + "=" * 60)
    log("Step 5: Feature importance...")
    log("=" * 60)
    
    try:
        plot_feature_importance(fitted_model, importance_type='split')
        plot_feature_importance(fitted_model, importance_type='gain')
    except Exception as e:
        log(f'Warning: Could not plot feature importance: {e}')
    
    log("\n" + "=" * 60)
    log("Training completed!")
    log("=" * 60)
    
    return fitted_model, eval_results, submission


if __name__ == '__main__':
    main()

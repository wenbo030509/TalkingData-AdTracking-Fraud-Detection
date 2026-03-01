"""
模型模块
包含LightGBM模型定义、训练、评估等功能
"""

from typing import Dict, Tuple

import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report

from config import (
    CATEGORICAL_FEATURES, SCALE_POS_WEIGHT, FIT_PARAMS
)
from utils import log


def default_model() -> LGBMClassifier:
    """
    创建默认的LightGBM模型
    
    Returns:
        LightGBM分类器
    """
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
    """
    基线模型配置
    
    Returns:
        配置好的LightGBM分类器
    """
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


BASE_MODELS = {
    'gbdt_base_001': gbdt_base_001()
}

SEARCH_PARAMS = {
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
    
    if SCALE_POS_WEIGHT is not None:
        log(f'use global config: scale_pos_weight={SCALE_POS_WEIGHT}')
        lgb_model.set_params(scale_pos_weight=SCALE_POS_WEIGHT)
    else:
        unbalance_degree = y_value_counts[majority_val] / y_value_counts[minority_val]
        log(f'majority_count/minority_count={unbalance_degree}')
        if unbalance_degree > 1.5 or unbalance_degree < 0.66:
            log('set is_unbalance=True')
            lgb_model.set_params(is_unbalance=True)
    
    return lgb_model


def get_model_and_search_params(base_model_id: str, search_params_id: str) -> Tuple[LGBMClassifier, Dict]:
    """
    获取模型和搜索参数
    
    Args:
        base_model_id: 基线模型ID
        search_params_id: 搜索参数ID
    
    Returns:
        (模型, 搜索参数字典)
    """
    return BASE_MODELS[base_model_id], SEARCH_PARAMS[search_params_id]


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
    from utils import g
    
    cat_fea_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
    
    log('------ fit parameters: ---------------------')
    log(f"early_stopping_rounds: {FIT_PARAMS['early_stopping_rounds']}")
    log(f"verbose: {FIT_PARAMS['verbose']}")
    log(f"eval_metric: {FIT_PARAMS['eval_metric']}")
    log(f'categorical_feature: {cat_fea_indices}')
    for i in cat_fea_indices:
        log(f"\t{i}: {CATEGORICAL_FEATURES[i]}")
    
    log('------ update balancing parameters: ---------------------')
    lgb_model = update_data_balancing_param(lgb_model, y_train.value_counts())
    
    log('------ model parameters: ---------------------')
    log(lgb_model.get_params())
    
    fitted = lgb_model.fit(
        X_train, y_train,
        categorical_feature=cat_fea_indices,
        early_stopping_rounds=FIT_PARAMS['early_stopping_rounds'],
        verbose=FIT_PARAMS['verbose'],
        eval_metric=FIT_PARAMS['eval_metric'],
        eval_set=[(X_validate, y_validate)]
    )
    
    g()
    return fitted


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
    y_pred_proba = model.predict_proba(X_vldt)[:, 1]
    y_pred_label = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_vldt, y_pred_proba)
    
    log(f'AUC: {auc:.6f}')
    log('\nClassification Report:')
    log(classification_report(y_vldt, y_pred_label, target_names=['not_attributed', 'is_attributed']))
    
    return {
        'auc': auc,
        'y_pred_proba': y_pred_proba,
        'y_pred_label': y_pred_label
    }


def plot_feature_importance(model: LGBMClassifier, importance_type: str = 'split', save_path: str = None):
    """
    绘制特征重要性
    
    Args:
        model: 训练好的模型
        importance_type: 重要性类型（'split'或'gain'）
        save_path: 保存路径
    """
    from config import OUTPUT_DIR
    
    if save_path is None:
        save_path = f'{OUTPUT_DIR}/feature_importance_{importance_type}.png'
    
    import matplotlib.pyplot as plt
    
    lgb.plot_importance(model.booster_, importance_type=importance_type, figsize=(10, 8))
    plt.title(f'Feature Importance ({importance_type})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

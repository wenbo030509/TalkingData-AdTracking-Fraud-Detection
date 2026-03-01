"""
TalkingData AdTracking Fraud Detection
======================================
基于LightGBM的广告欺诈检测模型

任务目标：预测用户点击广告后是否会下载安装APP（is_attributed=True）
核心挑战：
1. 极端不平衡数据集（正例/负例 ≈ 0.03/1）
2. 内存限制（数据集过大）
3. 验证集需要保持原始分布以反映真实效果

项目结构：
- config.py: 全局配置
- utils.py: 工具函数
- data_loader.py: 数据加载与预处理
- feature_engineering.py: 特征工程
- data_processor.py: 数据处理（IP桶、数据集合并）
- model.py: 模型定义、训练、评估
- main.py: 主入口

"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from config import OUTPUT_DIR
from data_processor import prep_feature_target_full_data
from model import BASE_MODELS, fit_model, evaluate_model, plot_feature_importance
from utils import log, timer_memory


def main():
    """
    主函数：完整的训练和预测流程
    """
    log("=" * 60)
    log("Step 1: Preparing data...")
    log("=" * 60)
    
    with timer_memory('Data preparation'):
        X_train, y_train, X_vldt, y_vldt, df_test = prep_feature_target_full_data()
    
    log("\n" + "=" * 60)
    log("Step 2: Training model...")
    log("=" * 60)
    
    base_model = BASE_MODELS['gbdt_base_001']
    
    with timer_memory('Model training'):
        fitted_model = fit_model(X_train, y_train, X_vldt, y_vldt, base_model)
    
    log("\n" + "=" * 60)
    log("Step 3: Evaluating model...")
    log("=" * 60)
    
    eval_results = evaluate_model(fitted_model, X_vldt, y_vldt)
    
    log("\n" + "=" * 60)
    log("Step 4: Generating predictions...")
    log("=" * 60)
    
    y_test_pred_proba = fitted_model.predict_proba(df_test)[:, 1]
    
    submission = pd.DataFrame({
        'click_id': df_test.index,
        'is_attributed': y_test_pred_proba
    })
    
    submission_path = f'{OUTPUT_DIR}/submission.csv'
    submission.to_csv(submission_path, index=False)
    log(f'Submission saved to: {submission_path}')
    log(f'Submission shape: {submission.shape}')
    log(f'Submission head:\n{submission.head()}')
    
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

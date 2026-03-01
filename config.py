"""
全局配置模块
包含所有全局配置参数
"""

# 数据路径配置
TRAIN_BUCKET_DIR = '/kaggle/input/train'
TEST_BUCKET_DIR = '/kaggle/input/test'
OUTPUT_DIR = '/kaggle/output'

# 数据处理配置
IP_BUCKET_NUM = 20                    # IP分桶数量
IS_DOWN_SAMPLE = True                  # 是否对训练集降采样
MAJORITY_MULTIPLY = 1                 # 多数类采样倍数（正例:负例 = 1:majority_multiply）
VLDT_SET_SIZE = 5000000               # 验证集总大小
SCALE_POS_WEIGHT = None                # LightGBM的类别权重（None表示自动计算）

# 类别特征列表
CATEGORICAL_FEATURES = ['app', 'device', 'os', 'channel', 'hh']

# 非训练列（需要删除的列）
NON_TRAIN_COLUMNS = ['click_time', 'dd', 'ip']

# 日志开关
ENABLE_LOG = True

# LightGBM训练参数
FIT_PARAMS = {
    'early_stopping_rounds': 25,
    'verbose': 10,
    'eval_metric': 'auc'
}

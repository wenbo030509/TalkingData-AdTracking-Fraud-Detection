# TalkingData AdTracking Fraud Detection 

## 项目概述
https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection
### 任务描述
Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China's largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user's click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you're challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. To support your modeling, they have provided a generous dataset covering approximately 200 million clicks over 4 days!

### Evaluation
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

任务目标：预测用户点击广告后是否会下载安装APP（is_attributed=True）

## 项目结构

```
.
├── Project_Process_and_Results_Documentation.md  # 项目过程与结果文档
├── README.md                # 项目说明文档
├── config.py                  # 全局配置模块
├── utils.py                   # 工具函数模块
├── data_loader.py             # 数据加载与预处理模块
├── feature_engineering.py      # 特征工程模块
├── data_processor.py          # 数据处理模块（IP桶、数据集合并）
├── model.py                  # 模型模块（定义、训练、评估）
├── main.py                   # 主入口
├── requirements.txt           # 依赖包列表
└── talkingdata_fraud_detection.py  # 完整的 fraud detection 脚本
```

## 模块说明

### 1. config.py - 全局配置
- 数据路径配置
- 数据处理参数（IP分桶数、降采样参数等）
- 类别特征列表
- LightGBM训练参数

### 2. utils.py - 工具函数
- `log()`: 日志输出
- `g()`: 垃圾回收
- `delete()`: 删除对象并回收内存
- `timer_memory()`: 计时器和内存监控
- `robust_boxcox()`: 稳健的Box-Cox变换
- `robust_inv_boxcox()`: Box-Cox逆变换

### 3. data_loader.py - 数据加载与预处理
- `read_data_file()`: 读取CSV文件
- `random_down_sample()`: 随机降采样
- `feature_target_split()`: 分割特征和目标变量

### 4. feature_engineering.py - 特征工程
- `extract_datetime_features()`: 提取时间特征（日期、小时）
- `extract_count_features()`: 提取计数统计特征（6个）
- `extract_nunique_features()`: 提取去重计数特征（5个）
- `extract_next_interval_features()`: 提取下次点击间隔特征（3个）
- `extract_cumcount_features()`: 提取累积计数特征（1个）
- `add_features()`: 添加所有特征

### 5. data_processor.py - 数据处理
- `process_ip_bucket()`: 处理单个IP桶的数据
- `prep_data_set_full_data()`: 准备完整数据集（合并所有IP桶）
- `prep_feature_target_full_data()`: 准备特征和目标变量

### 6. model.py - 模型
- `default_model()`: 创建默认LightGBM模型
- `gbdt_base_001()`: 基线模型配置
- `BASE_MODELS`: 基线模型字典
- `SEARCH_PARAMS`: 网格搜索参数字典
- `update_data_balancing_param()`: 更新数据平衡参数
- `get_model_and_search_params()`: 获取模型和搜索参数
- `fit_model()`: 训练LightGBM模型
- `evaluate_model()`: 评估模型性能
- `plot_feature_importance()`: 绘制特征重要性

### 7. main.py - 主入口
- `main()`: 完整的训练和预测流程

### 8. talkingdata_fraud_detection.py - 完整的 fraud detection 脚本
- 包含完整的广告欺诈检测功能，集成了所有模块的功能
- 独立运行的完整脚本，包含数据加载、特征工程、模型训练和评估等所有步骤

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `config.py` 文件，修改相关配置：

```python
# 数据路径配置
TRAIN_BUCKET_DIR = '/path/to/train'
TEST_BUCKET_DIR = '/path/to/test'
OUTPUT_DIR = '/path/to/output'

# 数据处理配置
IP_BUCKET_NUM = 20
IS_DOWN_SAMPLE = True
MAJORITY_MULTIPLY = 1
```

### 3. 运行训练

#### 方法1: 使用模块化脚本

```bash
python main.py
```

#### 方法2: 使用完整脚本

```bash
python talkingdata_fraud_detection.py
```

## 模块依赖关系

```
main.py
├── config.py
├── data_processor.py
│   ├── config.py
│   ├── data_loader.py
│   │   └── utils.py
│   └── feature_engineering.py
│       └── utils.py
└── model.py
    ├── config.py
    └── utils.py

talkingdata_fraud_detection.py
├── 独立脚本，包含所有功能模块
```

## 特征列表

### 时间特征
- `dd`: 点击日期
- `hh`: 点击小时

### 计数特征
- `cnt_grp_by_ip_device`: IP-Device组合的点击次数
- `cnt_grp_by_ip_app`: IP-App组合的点击次数
- `cnt_grp_by_ip_hh_app`: IP-小时-App组合的点击次数
- `cnt_grp_by_ip_hh_device`: IP-小时-Device组合的点击次数
- `cnt_grp_by_app_channel`: App-Channel组合的点击次数
- `cnt_grp_by_dd_hh_app_channel`: 日期-小时-App-Channel组合的点击次数

### 去重计数特征
- `nunique_on_channel_grp_by_ip`: 同一IP下的Channel去重数
- `nunique_on_app_grp_by_ip`: 同一IP下的App去重数
- `nunique_on_app_grp_by_ip_hh`: 同一IP-小时下的App去重数
- `nunique_on_channel_grp_by_app`: 同一App下的Channel去重数
- `nunique_on_channel_grp_by_hh_app`: 同一小时-App下的Channel去重数

### 时间间隔特征
- `nxt_itvl_by_ip_channel`: 同一IP-Channel的下次点击间隔
- `nxt_itvl_by_ip_app_channel`: 同一IP-App-Channel的下次点击间隔
- `nxt_itvl_by_ip_os_device_app`: 同一IP-OS-Device-App的下次点击间隔

### 累积计数特征
- `cumcount_on_app_grp_by_ip_device_os`: 同一IP-Device-OS下App的累积计数

## 扩展指南

### 添加新特征

在 `feature_engineering.py` 中添加新的特征提取函数：

```python
def extract_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    """提取新特征"""
    df['new_feature'] = df.groupby(['group_col'])['target_col'].transform('count')
    return df
```

然后在 `add_features()` 函数中调用：

```python
def add_features(df: pd.DataFrame, save_transformer: str = 'new_and_save') -> pd.DataFrame:
    df = extract_new_feature(df)
    return df
```

### 添加新模型

在 `model.py` 中添加新的模型配置：

```python
def new_model() -> LGBMClassifier:
    """新模型配置"""
    return default_model().set_params(
        learning_rate=0.1,
        num_leaves=120,
        # 其他参数...
    )

BASE_MODELS = {
    'gbdt_base_001': gbdt_base_001(),
    'new_model': new_model()
}
```

### 修改数据处理流程

在 `data_processor.py` 中修改 `process_ip_bucket()` 或 `prep_data_set_full_data()` 函数。

## 性能优化

1. **内存优化**：
   - 使用 `delete()` 及时释放内存
   - 使用 `g()` 执行垃圾回收
   - 分桶处理，避免一次性加载全部数据

2. **训练优化**：
   - 使用降采样加速训练
   - 使用Early Stopping防止过拟合
   - 使用类别特征减少内存占用

## 注意事项

1. 确保数据文件路径正确
2. 确保输出目录存在且有写权限
3. 根据实际内存情况调整 `IP_BUCKET_NUM`
4. 根据数据分布调整 `MAJORITY_MULTIPLY`

## 参考资料

- [Kaggle Competition Page](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TalkingData - 中国领先的大数据服务提供商](https://www.talkingdata.com/)

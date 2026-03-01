# TalkingData AdTracking Fraud Detection - 项目过程与结果文档

## 一、项目背景

### 1.1 竞赛信息
- **竞赛名称**: TalkingData AdTracking Fraud Detection
- **竞赛平台**: Kaggle
- **任务目标**: 预测用户点击广告后是否会下载安装APP（`is_attributed=True`）
- **评估指标**: AUC（Area Under the ROC Curve）

### 1.2 核心挑战

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| **极端不平衡数据集** | 正例/负例比例约为 0.03/1（约1:33） | 对训练集进行随机降采样，使正负样本比例均衡 |
| **内存限制** | 原始数据集过大，无法一次性加载到内存 | 采用IP分桶策略，将数据分成20个桶分别处理 |
| **模型评估** | 降采样验证集分数不能反映真实生产效果 | 验证集使用未降采样的原始数据 |
| **特征一致性** | 测试集仅包含3小时数据片段 | 剔除天级特征，仅使用小时级和多日汇总特征 |

## 二、数据探索

### 2.1 数据文件结构

```
/kaggle/input/talkingdata-adtracking-fraud-detection/
├── train.csv              # 训练集（按click_time排序）
├── test.csv               # 测试集（按click_time排序）
├── train_sample.csv       # 训练集样本
├── test_supplement.csv    # 测试集补充数据
└── sample_submission.csv  # 提交样例
```

### 2.2 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `ip` | int | IP地址 |
| `app` | int | APP ID |
| `device` | int | 设备ID |
| `os` | int | 操作系统ID |
| `channel` | int | 广告渠道ID |
| `click_time` | datetime | 点击时间 |
| `attributed_time` | datetime | 转化时间（仅训练集） |
| `is_attributed` | bool | 是否转化（目标变量） |
| `click_id` | int | 点击ID（仅测试集） |

### 2.3 数据分桶策略

为解决内存限制问题，采用**IP分桶策略**：

1. **分桶规则**: 按照 `IP % 20` 将数据集分成20个桶
2. **桶命名**: `train_0.csv` ~ `train_19.csv`, `test_0.csv` ~ `test_19.csv`
3. **分桶优势**:
   - 每个桶足够小，可以加载到内存
   - 可以独立进行特征工程，无需担心内存溢出
   - 便于并行处理

**分桶后数据大小示例**:
```
train_0.csv: 40M
train_1.csv: 40M
...
test_0.csv: 40M
test_1.csv: 40M
...
```

## 三、特征工程

### 3.1 特征工程总览

共提取 **15个统计特征**，分为5大类：

| 特征类别 | 数量 | 特征示例 |
|----------|------|----------|
| 时间特征 | 2 | `dd`, `hh` |
| 计数特征 | 6 | `cnt_grp_by_ip_device` 等 |
| 去重计数特征 | 5 | `nunique_on_channel_grp_by_ip` 等 |
| 时间间隔特征 | 3 | `nxt_itvl_by_ip_channel` 等 |
| 累积计数特征 | 1 | `cumcount_on_app_grp_by_ip_device_os` |

### 3.2 时间特征

```python
# 提取日期和小时
df['dd'] = df['click_time'].dt.day
df['hh'] = df['click_time'].dt.hour
```

### 3.3 计数特征（Count Features）

基于不同维度组合的点击次数统计：

| 特征名 | 分组维度 | 统计方式 |
|--------|----------|----------|
| `cnt_grp_by_ip_device` | IP + Device | count |
| `cnt_grp_by_ip_app` | IP + App | count |
| `cnt_grp_by_ip_hh_app` | IP + Hour + App | count |
| `cnt_grp_by_ip_hh_device` | IP + Hour + Device | count |
| `cnt_grp_by_app_channel` | App + Channel | count |
| `cnt_grp_by_dd_hh_app_channel` | Day + Hour + App + Channel | count |

**示例统计结果**:
```
cnt_grp_by_ip_device: max=241630; min=1; mean=20373.16
cnt_grp_by_ip_app: max=48182; min=1; mean=2442.11
cnt_grp_by_ip_hh_app: max=5415; min=1; mean=142.15
```

### 3.4 去重计数特征（Nunique Features）

统计不同维度下的唯一值数量：

| 特征名 | 分组维度 | 去重字段 |
|--------|----------|----------|
| `nunique_on_channel_grp_by_ip` | IP | Channel |
| `nunique_on_app_grp_by_ip` | IP | App |
| `nunique_on_app_grp_by_ip_hh` | IP + Hour | App |
| `nunique_on_channel_grp_by_app` | App | Channel |
| `nunique_on_channel_grp_by_hh_app` | Hour + App | Channel |

**示例统计结果**:
```
nunique_on_channel_grp_by_ip: max=151; min=1; mean=113.74
nunique_on_app_grp_by_ip: max=177; min=1; mean=63.01
nunique_on_app_grp_by_ip_hh: max=79; min=1; mean=27.48
```

### 3.5 时间间隔特征（Next Interval Features）

计算同一组合下的下次点击时间间隔（秒）：

| 特征名 | 分组维度 |
|--------|----------|
| `nxt_itvl_by_ip_channel` | IP + Channel |
| `nxt_itvl_by_ip_app_channel` | IP + App + Channel |
| `nxt_itvl_by_ip_os_device_app` | IP + OS + Device + App |

**计算方法**:
```python
df['nxt_itvl_by_ip_channel'] = df.groupby(['ip', 'channel'])['click_time'].diff().shift(-1).dt.seconds
```

**示例统计结果**:
```
nxt_itvl_by_ip_channel: max=341302; min=0; mean=4507.27
nxt_itvl_by_ip_app_channel: max=340696; min=0; mean=6990.01
nxt_itvl_by_ip_os_device_app: max=340928; min=0; mean=9497.46
```

**缺失值处理**: 使用均值填充
```python
fill_value = df['nxt_itvl_by_ip_channel'].mean()
df['nxt_itvl_by_ip_channel'].fillna(fill_value, inplace=True)
```

### 3.6 累积计数特征（Cumcount Features）

```python
# 按IP-Device-OS分组，App的累积计数
df['cumcount_on_app_grp_by_ip_device_os'] = df.groupby(['ip', 'device', 'os']).cumcount()
```

**示例统计结果**:
```
cumcount_on_app_grp_by_ip_device_os: max=59029; min=0; mean=1233.68
```

### 3.7 特征选择策略

**剔除天级特征的原因**:
- 训练集包含多天的数据
- 测试集仅包含3个小时的数据片段（2017-11-10 04:00:00）
- 使用天级特征会导致训练集与测试集分布不一致
- 仅保留小时级特征和多日汇总特征

## 四、数据预处理

### 4.1 降采样策略

**降采样原因**:
- 原始数据集极度不平衡（正例/负例 ≈ 1:33）
- 多数类样本过多，导致训练缓慢
- 降采样可以加速训练，同时保持模型性能

**降采样方法**:
```python
def random_down_sample(df, majority_multiply=1):
    """
    随机降采样
    majority_multiply: 多数类相对于少数类的倍数
    """
    df_minority = df[df['is_attributed'] == 1]  # 正例（少数类）
    df_majority = df[df['is_attributed'] == 0]  # 负例（多数类）
    
    minority_count = len(df_minority)
    majority_sample_size = minority_count * majority_multiply
    
    # 随机采样多数类
    df_majority_sampled = df_majority.sample(n=majority_sample_size, random_state=42)
    
    # 合并并打乱顺序
    df_downsampled = pd.concat([df_minority, df_majority_sampled])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42)
    
    return df_downsampled
```

**降采样效果**:
- 训练集降采样后：正例:负例 = 1:1
- 验证集：保持原始分布（不降采样）
- 测试集：保持原始分布（不降采样）

### 4.2 数据分割策略

**每个IP桶的数据分割**:
```python
# 训练集:验证集 = 4:1（从原始训练数据中分割）
vldt_len = min(g_vldt_set_size // g_ip_bkt_num, train_vldt_len // 5)

df_train = df_full[:train_vldt_len - vldt_len]   # 训练集（可降采样）
df_vldt = df_full[train_vldt_len - vldt_len:train_vldt_len]  # 验证集（不降采样）
df_test = df_full[train_vldt_len:]  # 测试集
```

**整体数据集规模**:
```
训练集（降采样后）: ~90万样本
验证集: 500万样本
测试集: ~1800万样本
```

### 4.3 类别特征处理

**类别特征列表**:
```python
g_categorical_features = ['app', 'device', 'os', 'channel', 'hh']
```

**LightGBM类别特征处理**:
```python
# 获取类别特征索引
cat_fea_indices = [X_train.columns.get_loc(col) for col in g_categorical_features]

# 传入LightGBM
model.fit(
    X_train, y_train,
    categorical_feature=cat_fea_indices,
    ...
)
```

## 五、模型训练

### 5.1 模型选择

**算法**: LightGBM (Light Gradient Boosting Machine)

**选择原因**:
- 训练速度快，内存占用低
- 支持类别特征直接输入
- 对不平衡数据处理效果好
- 支持Early Stopping防止过拟合

### 5.2 基线模型参数

```python
params = {
    'objective': 'binary',           # 二分类任务
    'metric': 'auc',                 # 评估指标：AUC
    'boosting_type': 'gbdt',         # 梯度提升决策树
    'learning_rate': 0.08,           # 学习率
    'num_leaves': 105,               # 叶子节点数
    'max_depth': 7,                  # 树的最大深度
    'min_split_gain': 0.3,           # 分裂最小增益
    'subsample': 0.8,                # 样本采样比例
    'subsample_freq': 1,             # 采样频率
    'colsample_bytree': 0.8,         # 特征采样比例
    'reg_alpha': 0.3,                # L1正则化
    'max_bin': 255,                  # 最大分箱数
    'n_estimators': 2500,            # 最大迭代次数
    'nthread': 4,                    # 线程数
    'verbose': 1,
    'iid': False,
    'two_round': True
}
```

### 5.3 训练参数

```python
g_fit_params = {
    'early_stopping_rounds': 25,     # 早停轮数
    'verbose': 10,                   # 每10轮输出日志
    'eval_metric': 'auc'             # 评估指标
}
```

### 5.4 数据平衡处理

```python
def update_data_balancing_param(lgb_model, y_value_counts):
    """根据数据分布自动设置平衡参数"""
    unbalance_degree = y_value_counts[0] / y_value_counts[1]
    
    if unbalance_degree > 1.5 or unbalance_degree < 0.66:
        # 数据不平衡，设置is_unbalance=True
        lgb_model.set_params(is_unbalance=True)
    
    return lgb_model
```

### 5.5 网格搜索实验

**实验设计**:

| 实验ID | 搜索参数 | 参数值 |
|--------|----------|--------|
| `gbdt_base_001_exp_002` | `min_sum_hessian_in_leaf` | [0.001, 0.01, 0.05, 0.1] |
| `gbdt_base_001_exp_005` | `min_split_gain` | [0.3, 0.4, 0.5] |
| `gbdt_base_001_exp_010` | `num_leaves` + `min_split_gain` | [90,105] × [0.3,0.4] |

**网格搜索结果示例**:

**Experiment 002 - min_sum_hessian_in_leaf**:

| 参数值 | Precision | Recall | F1 | AUC |
|--------|-----------|--------|-----|-----|
| 0.001 | 0.087599 | 0.928669 | 0.160097 | 0.955150 |
| 0.01 | 0.087599 | 0.928669 | 0.160097 | 0.955150 |
| 0.05 | 0.088253 | 0.928142 | 0.161180 | 0.954966 |
| **0.1** | **0.090201** | **0.928142** | **0.164422** | **0.955182** |

**Experiment 005 - min_split_gain**:

| 参数值 | Precision | Recall | F1 | AUC |
|--------|-----------|--------|-----|-----|
| 0.3 | 0.090201 | 0.928142 | 0.164422 | 0.955182 |
| **0.4** | **0.087830** | **0.929725** | **0.160498** | **0.955694** |
| 0.5 | 0.084776 | 0.929830 | 0.155384 | 0.955384 |

**Experiment 010 - 组合搜索**:

| num_leaves | min_split_gain | Precision | Recall | F1 | AUC |
|------------|----------------|-----------|--------|-----|-----|
| 90 | 0.3 | 0.087663 | 0.928986 | 0.160208 | 0.955313 |
| 105 | 0.3 | 0.090201 | 0.928142 | 0.164422 | 0.955182 |
| 90 | 0.4 | 0.089005 | 0.929197 | 0.162450 | 0.955568 |
| **105** | **0.4** | **0.087830** | **0.929725** | **0.160498** | **0.955694** |

**最优参数组合**:
```python
best_params = {
    'num_leaves': 105,
    'min_split_gain': 0.4,
    'min_sum_hessian_in_leaf': 0.1
}
```

## 六、模型评估

### 6.1 评估指标

**主要指标**: AUC (Area Under the ROC Curve)

**辅助指标**:
- Precision（精确率）
- Recall（召回率）
- F1-Score
- Confusion Matrix（混淆矩阵）

### 6.2 验证集性能

**分类报告**:

```
                   precision    recall  f1-score   support

is_not_attributed       1.00      0.98      0.99   4990523
    is_attributed       0.09      0.93      0.16      9477

         accuracy                           0.98   5000000
        macro avg       0.54      0.96      0.58   5000000
     weighted avg       1.00      0.98      0.99   5000000
```

**关键指标**:
- **AUC**: 0.955694
- **正例召回率**: 0.93（能够识别93%的欺诈点击）
- **正例精确率**: 0.09（由于数据极度不平衡，精确率较低是正常的）

### 6.3 混淆矩阵

| 预测\实际 | 负例 (0) | 正例 (1) |
|-----------|----------|----------|
| 负例 (0) | 4,890,546 | 663 |
| 正例 (1) | 99,977 | 8,814 |

**分析**:
- True Negative: 4,890,546（正确识别非欺诈点击）
- False Positive: 99,977（误报）
- False Negative: 663（漏报）
- True Positive: 8,814（正确识别欺诈点击）

### 6.4 ROC曲线和PR曲线

**ROC曲线分析**:
- 曲线接近左上角，说明模型区分能力强
- AUC = 0.955694，表明模型性能优秀

**PR曲线分析**:
- 由于数据不平衡，PR曲线更能反映模型性能
- 曲线下的面积较大，说明模型在正例识别上表现良好

## 七、特征重要性分析

### 7.1 特征重要性排名（Split）

根据LightGBM的特征重要性（split类型）：

| 排名 | 特征名 | 重要性 |
|------|--------|--------|
| 1 | `nxt_itvl_by_ip_os_device_app` | 最高 |
| 2 | `cnt_grp_by_ip_device` | 高 |
| 3 | `nxt_itvl_by_ip_app_channel` | 高 |
| 4 | `nxt_itvl_by_ip_channel` | 高 |
| 5 | `cnt_grp_by_ip_app` | 中高 |
| 6 | `cumcount_on_app_grp_by_ip_device_os` | 中高 |
| 7 | `cnt_grp_by_app_channel` | 中 |
| 8 | `nunique_on_channel_grp_by_ip` | 中 |
| 9 | `nunique_on_app_grp_by_ip` | 中 |
| 10 | `app` | 中 |

### 7.2 关键特征解读

**Top 1: `nxt_itvl_by_ip_os_device_app`**
- 同一IP-OS-Device-App组合的下次点击间隔
- 欺诈点击通常具有规律性的时间间隔模式
- 正常用户的点击间隔更加随机

**Top 2: `cnt_grp_by_ip_device`**
- 同一IP-Device组合的点击次数
- 高点击次数可能表明机器人行为

**Top 3: `nxt_itvl_by_ip_app_channel`**
- 同一IP-App-Channel组合的下次点击间隔
- 反映用户在特定渠道对特定APP的点击模式

## 八、实验管理与参数追溯

### 8.1 基线模型管理

```python
g_base_models = {
    'gbdt_base_001': gbdt_base_001()
}
```

### 8.2 实验参数管理

```python
g_search_params = {
    'gbdt_base_001_exp_002': {'min_sum_hessian_in_leaf': [0.001, 0.01, 0.05, 0.1]},
    'gbdt_base_001_exp_005': {'min_split_gain': [0.3, 0.4, 0.5]},
    'gbdt_base_001_exp_010': {'num_leaves': [90, 105], 'min_split_gain': [0.3, 0.4]}
}
```

### 8.3 命名规范

- **基线模型**: `{algorithm}_base_{version}`，如 `gbdt_base_001`
- **实验**: `{base_model_id}_exp_{experiment_number}`，如 `gbdt_base_001_exp_002`

## 九、提交结果

### 9.1 预测生成

```python
# 预测测试集
y_test_pred_proba = fitted_model.predict_proba(df_test)[:, 1]

# 生成提交文件
submission = pd.DataFrame({
    'click_id': df_test.index,
    'is_attributed': y_test_pred_proba
})

submission.to_csv('submission.csv', index=False)
```

### 9.2 提交文件格式

```csv
click_id,is_attributed
0,0.123456
1,0.234567
2,0.345678
...
```

## 十、总结与反思

### 10.1 关键成功因素

1. **IP分桶策略**: 有效解决内存限制问题，使大规模数据处理成为可能
2. **合理的降采样**: 在保证模型性能的前提下，大幅提升训练速度
3. **特征工程**: 时间间隔特征和计数特征对识别欺诈点击非常有效
4. **验证集设计**: 使用未降采样的原始数据作为验证集，确保评估结果反映真实性能
5. **参数管理**: 系统化的实验管理，便于追溯和复现

### 10.2 可改进方向

1. **特征工程**:
   - 添加更多时间窗口统计特征（如过去1小时、6小时的点击次数）
   - 添加用户行为序列特征
   - 使用Embedding编码高基数类别特征

2. **模型优化**:
   - 尝试XGBoost、CatBoost等其他GBDT算法
   - 使用交叉验证进行更稳健的模型评估
   - 尝试模型融合（Ensemble）

3. **数据处理**:
   - 使用更复杂的降采样策略（如SMOTE）
   - 对时间间隔特征进行更精细的分箱处理

### 10.3 技术亮点

1. **内存管理**: 通过分桶和及时垃圾回收，在有限内存下处理大规模数据
2. **不平衡数据处理**: 结合降采样和LightGBM的is_unbalance参数，有效处理极端不平衡数据
3. **特征一致性**: 剔除天级特征，确保训练集与测试集分布一致
4. **实验可复现**: 完整的参数管理和日志记录，确保实验可复现

## 十一、附录

### 11.1 完整特征列表

| 序号 | 特征名 | 类型 | 说明 |
|------|--------|------|------|
| 1 | `app` | 类别 | APP ID |
| 2 | `device` | 类别 | 设备ID |
| 3 | `os` | 类别 | 操作系统ID |
| 4 | `channel` | 类别 | 广告渠道ID |
| 5 | `hh` | 类别 | 点击小时 |
| 6 | `nxt_itvl_by_ip_os_device_app` | 数值 | IP-OS-Device-App下次点击间隔 |
| 7 | `cnt_grp_by_dd_hh_app_channel` | 数值 | 日期-小时-App-Channel点击次数 |
| 8 | `cumcount_on_app_grp_by_ip_device_os` | 数值 | IP-Device-OS下App累积计数 |
| 9 | `cnt_grp_by_ip_device` | 数值 | IP-Device点击次数 |
| 10 | `nunique_on_channel_grp_by_ip` | 数值 | IP下Channel去重数 |
| 11 | `nunique_on_app_grp_by_ip` | 数值 | IP下App去重数 |
| 12 | `nxt_itvl_by_ip_channel` | 数值 | IP-Channel下次点击间隔 |
| 13 | `cnt_grp_by_ip_hh_device` | 数值 | IP-小时-Device点击次数 |
| 14 | `nxt_itvl_by_ip_app_channel` | 数值 | IP-App-Channel下次点击间隔 |
| 15 | `cnt_grp_by_app_channel` | 数值 | App-Channel点击次数 |
| 16 | `cnt_grp_by_ip_app` | 数值 | IP-App点击次数 |
| 17 | `nunique_on_app_grp_by_ip_hh` | 数值 | IP-小时下App去重数 |
| 18 | `cnt_grp_by_ip_hh_app` | 数值 | IP-小时-App点击次数 |
| 19 | `nunique_on_channel_grp_by_app` | 数值 | App下Channel去重数 |
| 20 | `nunique_on_channel_grp_by_hh_app` | 数值 | 小时-App下Channel去重数 |

### 11.2 参考文献

1. LightGBM Documentation: https://lightgbm.readthedocs.io/
2. Kaggle Competition: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
3. Ke, Guolin, et al. "LightGBM: A highly efficient gradient boosting decision tree." NeurIPS 2017.

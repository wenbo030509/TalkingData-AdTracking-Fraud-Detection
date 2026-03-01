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

---

## 十二、特征指标深度解读

### 12.1 特征设计核心思想

广告欺诈检测的核心是识别**异常点击行为**。正常用户和欺诈用户（机器人/作弊脚本）在行为模式上存在本质差异：

| 行为维度 | 正常用户 | 欺诈用户 |
|----------|----------|----------|
| **点击频率** | 随机、有间隔 | 高频、规律 |
| **点击多样性** | 多样（不同APP/渠道） | 单一、重复 |
| **时间模式** | 符合作息规律 | 24小时不间断 |
| **设备使用** | 固定几台设备 | 大量设备轮换 |

基于以上差异，特征工程从**频率、多样性、时间规律**三个维度构建指标体系。

---

### 12.2 基础类别特征（5个）

#### 12.2.1 `app` - APP ID
**含义**：用户点击的广告所属的应用ID

**为什么使用**：
- 不同APP的受众群体不同，某些APP可能更容易成为欺诈目标
- 某些APP与特定欺诈模式相关联
- 作为类别特征，LightGBM可以自动学习APP的Embedding表示

**欺诈检测价值**：
- 识别高风险APP（游戏类、工具类APP常被欺诈攻击）
- 发现APP与渠道的组合欺诈模式

---

#### 12.2.2 `device` - 设备ID
**含义**：用户使用的设备标识

**为什么使用**：
- 设备是识别用户身份的重要维度
- 同一设备多次点击可能是正常用户，也可能是设备农场

**欺诈检测价值**：
- 识别设备农场（大量点击来自少量设备）
- 发现设备与IP的不匹配（模拟器/虚拟机特征）

---

#### 12.2.3 `os` - 操作系统ID
**含义**：设备运行的操作系统

**为什么使用**：
- 不同OS用户行为模式不同
- 某些欺诈工具可能只针对特定OS

**欺诈检测价值**：
- 识别异常的OS分布（如大量点击来自老旧/罕见OS版本）
- 发现OS与设备的矛盾组合

---

#### 12.2.4 `channel` - 广告渠道ID
**含义**：广告投放的渠道标识

**为什么使用**：
- 不同渠道的质量差异很大
- 某些渠道可能存在流量作弊问题

**欺诈检测价值**：
- 识别高风险渠道
- 发现渠道与APP的异常关联

---

#### 12.2.5 `hh` - 点击小时
**含义**：点击发生的小时数（0-23）

**为什么使用**：
- 人类用户有明确的作息规律，机器人则24小时运行
- 测试集仅包含特定时间段数据，小时特征比天特征更稳定

**欺诈检测价值**：
- 识别非人类作息模式（如凌晨3-5点大量点击）
- 结合其他特征发现时间聚集性欺诈

---

### 12.3 计数特征（6个）

计数特征反映**点击频率和密度**，是识别高频欺诈的核心指标。

#### 12.3.1 `cnt_grp_by_ip_device` - IP-Device点击次数
**含义**：同一IP和设备组合下的总点击次数

**计算方法**：
```python
df['cnt_grp_by_ip_device'] = df.groupby(['ip', 'device'])['channel'].transform('count')
```

**为什么使用**：
- 正常用户通常使用固定设备，点击次数有限
- 欺诈脚本可能使用同一IP下的多台设备（设备农场）
- 该指标能识别IP-Device组合的异常高频点击

**欺诈检测价值**：
- **高值含义**：该IP-Device组合存在大量点击，可能是机器人或设备农场
- **阈值参考**：均值约20,373，最大值达241,630，远超正常用户行为

---

#### 12.3.2 `cnt_grp_by_ip_app` - IP-App点击次数
**含义**：同一IP对同一APP的点击次数

**为什么使用**：
- 正常用户对同一APP的点击次数有限
- 欺诈脚本可能针对特定APP进行刷量

**欺诈检测价值**：
- 识别针对特定APP的IP级攻击
- 发现IP与APP的异常关联（如某IP只点击某APP）

---

#### 12.3.3 `cnt_grp_by_ip_hh_app` - IP-小时-App点击次数
**含义**：同一IP在特定小时对特定APP的点击次数

**为什么使用**：
- 在小时粒度上识别点击聚集
- 欺诈行为往往在短时间内集中发生

**欺诈检测价值**：
- 识别短时高频攻击（如1小时内某IP对某APP点击数百次）
- 比日级粒度更精准地定位欺诈时段

---

#### 12.3.4 `cnt_grp_by_ip_hh_device` - IP-小时-Device点击次数
**含义**：同一IP-设备在特定小时的点击次数

**为什么使用**：
- 识别单设备在短时间内的异常行为
- 防止正常用户的多次有效点击被误判

**欺诈检测价值**：
- 区分"用户多次浏览"和"机器刷量"
- 小时粒度减少正常用户的误杀

---

#### 12.3.5 `cnt_grp_by_app_channel` - App-Channel点击次数
**含义**：特定APP在特定渠道的总点击次数

**为什么使用**：
- 某些APP-Channel组合可能存在系统性欺诈
- 反映渠道质量（高点击低转化=可疑）

**欺诈检测价值**：
- 识别问题渠道（某渠道对某APP的点击异常高）
- 发现APP-Channel的异常关联模式

---

#### 12.3.6 `cnt_grp_by_dd_hh_app_channel` - 日期-小时-App-Channel点击次数
**含义**：特定时间窗口内，特定APP-Channel的点击次数

**为什么使用**：
- 最细粒度的点击密度指标
- 能捕获精准的欺诈时间窗口

**欺诈检测价值**：
- 识别特定时段的集中攻击（如双11凌晨某渠道对某APP的刷量）
- 为运营团队提供精准的风控时间窗口

---

### 12.4 去重计数特征（5个）

去重计数特征反映**行为多样性**，正常用户行为多样，欺诈用户行为单一。

#### 12.4.1 `nunique_on_channel_grp_by_ip` - IP下Channel去重数
**含义**：同一IP点击的不同渠道数量

**计算方法**：
```python
df['nunique_on_channel_grp_by_ip'] = df.groupby('ip')['channel'].transform('nunique')
```

**为什么使用**：
- 正常用户会浏览多个渠道的广告
- 欺诈IP可能只针对特定渠道（如与作弊渠道合作）

**欺诈检测价值**：
- **低值含义**：IP只点击少数几个渠道，可能是渠道作弊
- **高值含义**：IP点击渠道多样，更可能是正常用户
- **均值参考**：约113.74，说明正常用户确实会接触多个渠道

---

#### 12.4.2 `nunique_on_app_grp_by_ip` - IP下App去重数
**含义**：同一IP点击的不同APP数量

**为什么使用**：
- 正常用户会关注多种类型的APP
- 欺诈脚本可能只针对特定APP刷量

**欺诈检测价值**：
- 识别单一APP攻击（某IP只点击某APP，极可能是刷量）
- 均值约63，说明正常用户确实会浏览多个APP

---

#### 12.4.3 `nunique_on_app_grp_by_ip_hh` - IP-小时下App去重数
**含义**：同一IP在特定小时内点击的不同APP数量

**为什么使用**：
- 在小时粒度上评估行为多样性
- 正常用户1小时内通常不会浏览太多APP

**欺诈检测价值**：
- **异常高值**：1小时内点击大量不同APP，可能是机器人遍历
- **异常低值**：1小时内只点击1个APP多次，可能是刷量

---

#### 12.4.4 `nunique_on_channel_grp_by_app` - App下Channel去重数
**含义**：特定APP在不同渠道的被点击数量

**为什么使用**：
- 反映APP的渠道分布策略
- 某些APP可能只在特定渠道投放

**欺诈检测价值**：
- 识别渠道作弊（某APP只在问题渠道有高点击）
- 发现APP的渠道依赖模式

---

#### 12.4.5 `nunique_on_channel_grp_by_hh_app` - 小时-App下Channel去重数
**含义**：特定小时-APP组合下，有多少不同渠道产生点击

**为什么使用**：
- 识别特定时段的渠道聚集
- 某些欺诈行为可能在特定时段集中发生

**欺诈检测价值**：
- 发现时段性渠道攻击（如凌晨某APP被多个可疑渠道点击）

---

### 12.5 时间间隔特征（3个）

时间间隔特征反映**点击规律性**，是识别机器人的最强指标。

#### 12.5.1 `nxt_itvl_by_ip_channel` - IP-Channel下次点击间隔
**含义**：同一IP在同一渠道的两次点击之间的时间间隔（秒）

**计算方法**：
```python
df['nxt_itvl_by_ip_channel'] = df.groupby(['ip', 'channel'])['click_time'].diff().shift(-1).dt.seconds
```

**为什么使用**：
- **最核心的欺诈检测指标**
- 人类点击间隔随机，机器人点击间隔规律（如固定5秒一次）
- 该特征在特征重要性中排名Top 3

**欺诈检测价值**：
- **规律间隔**：固定或相似的时间间隔是机器人的典型特征
- **均值参考**：约4507秒（75分钟），正常用户不会如此规律地点击
- **缺失值处理**：用均值填充，表示该组合的最后一次点击

---

#### 12.5.2 `nxt_itvl_by_ip_app_channel` - IP-App-Channel下次点击间隔
**含义**：同一IP对同一APP在同一渠道的两次点击间隔

**为什么使用**：
- 比IP-Channel更细粒度
- 能识别针对特定APP的刷量行为

**欺诈检测价值**：
- 识别APP级的规律点击
- 均值约6990秒，比IP-Channel间隔更长，说明用户对特定APP的重复点击更少

---

#### 12.5.3 `nxt_itvl_by_ip_os_device_app` - IP-OS-Device-App下次点击间隔
**含义**：同一IP-OS-Device-App组合的两次点击间隔

**为什么使用**：
- **特征重要性排名第一**
- 最细粒度的设备级时间间隔
- 能识别同一设备上对同一APP的规律操作

**欺诈检测价值**：
- **最强欺诈指标**：均值约9497秒，最大值340928秒（约4天）
- 正常用户不会用同一设备如此规律地点击同一APP
- 该特征能精准识别设备农场和模拟器

---

### 12.6 累积计数特征（1个）

#### 12.6.1 `cumcount_on_app_grp_by_ip_device_os` - IP-Device-OS下App累积计数
**含义**：同一IP-Device-OS组合下，当前APP是第几次被点击

**计算方法**：
```python
df['cumcount_on_app_grp_by_ip_device_os'] = df.groupby(['ip', 'device', 'os']).cumcount()
```

**为什么使用**：
- 反映用户在设备上的操作序列
- 能识别新设备/新IP的异常行为

**欺诈检测价值**：
- **低值含义**：新设备/IP的前几次点击，风险较高
- **高值含义**：老用户的常规操作，风险较低
- 均值约1233，最大值59029，可用于识别设备生命周期

---

### 12.7 特征重要性总结

根据LightGBM的特征重要性排名：

| 排名 | 特征名 | 重要性等级 | 核心检测能力 |
|------|--------|------------|--------------|
| 1 | `nxt_itvl_by_ip_os_device_app` | ★★★★★ | 设备级规律点击（最强） |
| 2 | `cnt_grp_by_ip_device` | ★★★★☆ | IP-Device高频点击 |
| 3 | `nxt_itvl_by_ip_app_channel` | ★★★★☆ | APP级规律点击 |
| 4 | `nxt_itvl_by_ip_channel` | ★★★★☆ | 渠道级规律点击 |
| 5 | `cnt_grp_by_ip_app` | ★★★☆☆ | IP-APP高频点击 |
| 6 | `cumcount_on_app_grp_by_ip_device_os` | ★★★☆☆ | 设备生命周期 |

**关键洞察**：
1. **时间间隔特征**占据Top 1/3/4，是识别机器人的最强武器
2. **计数特征**占据Top 2/5，用于识别高频攻击
3. **累积计数**反映设备历史，辅助判断新设备风险

---

### 12.8 特征设计方法论

#### 12.8.1 多粒度设计
特征从粗到细覆盖多个粒度：

```
粗粒度 ──────────────────────────────> 细粒度
IP ──> IP+Device ──> IP+OS+Device+App
     └──> IP+Channel ──> IP+App+Channel
```

**优势**：
- 粗粒度特征捕获宏观模式（IP级攻击）
- 细粒度特征捕获微观异常（设备级规律点击）
- 多粒度组合提升模型泛化能力

#### 12.8.2 时间维度设计
时间特征从大到小：

```
天(dd) ──> 小时(hh) ──> 秒级间隔
```

**注意**：由于测试集只有3小时数据，剔除`dd`特征，避免分布不一致。

#### 12.8.3 统计方式选择
| 统计方式 | 适用场景 | 本项目应用 |
|----------|----------|------------|
| count | 频率检测 | 6个计数特征 |
| nunique | 多样性检测 | 5个去重特征 |
| diff/shift | 规律性检测 | 3个时间间隔特征 |
| cumcount | 序列检测 | 1个累积特征 |

---

### 12.9 特征工程最佳实践

1. **业务理解优先**：先理解欺诈行为模式，再设计对应特征
2. **多粒度覆盖**：从粗到细多维度刻画行为
3. **分布一致性**：确保训练集与测试集特征分布一致
4. **重要性验证**：通过模型特征重要性验证设计有效性
5. **可解释性**：每个特征都应有明确的业务含义

### 11.2 参考文献

1. LightGBM Documentation: https://lightgbm.readthedocs.io/
2. Kaggle Competition: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
3. Ke, Guolin, et al. "LightGBM: A highly efficient gradient boosting decision tree." NeurIPS 2017.

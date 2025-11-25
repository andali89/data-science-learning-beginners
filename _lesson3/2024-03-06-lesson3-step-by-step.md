---
title: 2. 房屋价格预测:步骤详解与模型指南
author: Anda Li
date: 2024-03-06
category: Data Science Learning
layout: post
---

本指南是 [**房屋价格预测实验教学案例**]({{ site.baseurl }}/data-science-learning-beginners/lesson3/2024-03-06-lesson3-step-by-step/) 的配套技术文档。

在这里，我们将深入代码细节，一步步演示如何使用 Python (Pandas, Scikit-learn) 实现从数据清洗、探索性分析 (EDA) 到构建多元线性回归及高级模型 (CART, SVR, MLP) 的完整流程。请参照本指南完成代码编写，并将运行结果填入实验报告中。

> :gift: **配套资源**
>
> 本章涉及的完整代码和数据集可以通过链接 —— <a href="{{ site.baseurl }}/assets/code/lesson3/codes.zip" target="_blank" download="codes.zip">示例代码与数据 (Zip)</a> 下载。
{: .block-warning }

## 1. 实验环境准备 (Environment Setup)

**目的**：导入数据分析和建模所需的 Python 库。

**代码示例**：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

**详细说明**：
-   `numpy` (`np`): 提供高性能的多维数组对象及相关操作，是 Python 数值计算的基础。
-   `pandas` (`pd`): 提供 DataFrame 数据结构，用于高效地处理表格数据（读取、清洗、筛选等）。
-   `matplotlib.pyplot` (`plt`): Python 的基础绘图库，用于创建静态图表。
-   `seaborn` (`sns`): 基于 Matplotlib 的高级绘图库，提供更美观的统计图形接口。
-   `sklearn` (scikit-learn): 机器学习核心库。
    -   `preprocessing`: 包含数据预处理工具，如标准化 (`StandardScaler`)。
    -   `model_selection`: 包含数据集划分工具 (`train_test_split`)。
    -   `linear_model`: 包含线性模型算法 (`LinearRegression`)。
    -   `metrics`: 包含模型评估指标 (MAE, MSE, R2 等)。

## 2. 数据读取 (Data Loading)

**目的**：将 CSV 格式的数据集加载到 Pandas DataFrame 中以便分析。

**代码示例**：
```python
df = pd.read_csv('./house_price_regression_dataset.csv')
df.head()  # 查看前5行
df.info()  # 查看数据基本信息
```

**函数详解**：
-   `pd.read_csv(filepath)`: 读取逗号分隔值文件。
    -   `filepath`: 文件路径字符串。
-   `df.head(n)`: 返回 DataFrame 的前 `n` 行，默认为 5。用于快速检查数据加载是否正确。
-   `df.info()`: 打印 DataFrame 的简要摘要，包括索引类型、列名、非空值数量和数据类型。用于检查是否有缺失值及数据类型是否符合预期。

## 3. 数据的探索性分析 (Exploratory Data Analysis, EDA)

**目的**：通过统计和可视化手段了解数据的分布特征及变量间的关系。

### 3.1 统计描述
**代码示例**：
```python
df.describe()
```
**函数详解**：
-   `df.describe()`: 生成描述性统计数据。对于数值型列，计算计数、均值、标准差、最小值、25%/50%/75% 分位数和最大值。

### 3.2 数据可视化
**代码示例 (直方图与箱线图)**：
```python
plt.hist(df['House_Price'], bins=50) # 直方图
sns.boxplot(y=df['House_Price'])     # 箱线图
```
**函数详解**：
-   `plt.hist(x, bins)`: 绘制直方图，展示数据的频率分布。`bins` 参数控制直方图的柱子数量。
-   `sns.boxplot(y=data)`: 绘制箱线图，展示数据的五数概括（最小值、第一四分位数、中位数、第三四分位数、最大值）及异常值。

**代码示例 (散点图)**：
```python
plt.scatter(df['Square_Footage'], df['House_Price'])
```
**函数详解**：
-   `plt.scatter(x, y)`: 绘制散点图，用于观察两个数值变量之间的关系（如线性关系、聚类等）。

**代码示例 (相关性热力图)**：
```python
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```
**函数详解**：
-   `df.corr()`: 计算列之间的成对相关系数（默认为 Pearson 相关系数）。
-   `sns.heatmap(data, annot=True, cmap='coolwarm')`: 绘制热力图。
    -   `annot=True`: 在每个方格中显示数值。
    -   `cmap='coolwarm'`: 设置颜色映射，红色表示正相关，蓝色表示负相关。

## 4. 数据预处理 (Data Preprocessing)

**目的**：将原始数据转换为适合模型训练的格式。

### 4.1 特征工程
**代码示例**：
```python
df['House_Age'] = 2025 - df['Year_Built']
df = df.drop(columns='Year_Built')
```
**操作详解**：
-   创建新特征 `House_Age`（房龄），因为它比单纯的建造年份更能反映房屋的新旧程度对价格的影响。
-   `df.drop(columns='Year_Built')`: 删除原始的 `Year_Built` 列，避免特征冗余（多重共线性）。

### 4.2 提取特征与目标
**代码示例**：
```python
X = df.drop(columns='House_Price')
y = df['House_Price']
```
**操作详解**：
-   `X` (特征矩阵): 包含所有用于预测的变量。
-   `y` (目标向量): 包含我们要预测的变量（房价）。

### 4.3 数据标准化
**代码示例**：
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**函数详解**：
-   `StandardScaler()`: 标准化特征，通过去除均值并缩放到单位方差。公式为：$z = (x - u) / s$。
-   `fit_transform(X)`: 先计算 X 的均值和标准差（fit），然后对 X 进行转换（transform）。这对许多机器学习算法（如 SVM, MLP, 线性回归）至关重要，因为它们对特征的量纲敏感。

### 4.4 划分数据集
**代码示例**：
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)
```
**函数详解**：
-   `train_test_split(*arrays, test_size, random_state)`: 将数组或矩阵拆分为随机的训练子集和测试子集。
    -   `test_size=0.2`: 表示 20% 的数据用于测试，80% 用于训练。
    -   `random_state=123`: 随机数生成器的种子。设置此参数可确保每次运行代码时拆分结果相同，保证实验可复现。

## 5. 多元线性回归模型 (Linear Regression)

**代码示例**：
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
```
**函数详解**：
-   `LinearRegression()`: 创建一个普通最小二乘法线性回归对象。
-   `fit(X, y)`: 训练模型。模型会学习特征 X 和目标 y 之间的线性关系（即计算回归系数）。
-   `predict(X)`: 使用训练好的模型对新数据 X 进行预测。

**模型评估**：
```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```
-   **MAE**: 平均绝对误差，反映预测值偏离真实值的平均幅度。
-   **MSE**: 均方误差，对大误差更敏感。
-   **RMSE**: 均方根误差，量纲与原数据一致，易于解释。
-   **R²**: 决定系数，表示模型解释了目标变量方差的百分比。1 表示完美预测，0 表示模型不如直接取均值。

## 6. 高级回归模型 (Advanced Models)

在 `House Prices Prediction_Advanced_Models.ipynb` 中，我们使用了以下三种模型。代码结构与线性回归完全一致：**初始化 -> 训练 (fit) -> 预测 (predict) -> 评估**。

### 6.1 CART 决策树回归 (Decision Tree)
**代码示例**：
```python
from sklearn.tree import DecisionTreeRegressor
cart_model = DecisionTreeRegressor(random_state=123)
cart_model.fit(X_train, y_train)
```
**模型详解**：
-   **原理**：通过一系列“如果-那么”规则对数据空间进行划分。
-   **参数**：`random_state` 用于控制树构建过程中的随机性（如特征选择）。

### 6.2 SVR 支持向量回归 (Support Vector Regression)
**代码示例**：
```python
from sklearn.svm import SVR
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
```
**模型详解**：
-   **原理**：寻找一个超平面，使得大部分样本点落在该平面的间隔带内。
-   **参数**：`kernel='rbf'` 指定使用径向基函数（RBF）作为核函数，这使得模型能够处理非线性关系。

### 6.3 MLP 多层感知机 (Neural Network)
**代码示例**：
```python
from sklearn.neural_network import MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123)
mlp_model.fit(X_train, y_train)
```
**模型详解**：
-   **原理**：一种前馈人工神经网络，通过隐藏层学习数据的复杂非线性特征。
-   **参数**：
    -   `hidden_layer_sizes=(100, 50)`: 定义网络结构。这里有两个隐藏层，第一层 100 个神经元，第二层 50 个神经元。
    -   `max_iter=1000`: 最大迭代次数（训练轮数），确保模型有足够的时间收敛。
    -   `activation`: 激活函数，决定神经元如何处理输入信号。可选 `'relu'`（默认）, `'tanh'`, `'logistic'`。不同激活函数适用于不同类型的数据分布。

## 7. 参数调整指南 (Parameter Tuning Guide)

本节介绍实验中可以调整的关键参数，鼓励大家尝试不同的设置，观察其对模型性能的影响。

### 7.1 通用参数

*   **`random_state` (随机种子)**
    *   **位置**: `train_test_split`, `DecisionTreeRegressor`, `MLPRegressor` 等。
    *   **说明**: 控制随机过程的种子。
    *   **调整建议**: 修改数字（如 123 改为 42），观察模型结果是否发生显著变化。如果变化很大，说明模型可能不稳定或数据量不足。
*   **`test_size` (测试集比例)**
    *   **位置**: `train_test_split`。
    *   **说明**: 决定了多少数据用于评估模型。
    *   **调整建议**: 常用值在 0.2 到 0.3 之间。
        *   调大 (如 0.4): 训练数据减少，模型可能欠拟合，但评估结果可能更具代表性。
        *   调小 (如 0.1): 训练数据增加，模型可能学得更好，但评估结果可能波动较大。

### 7.2 模型特定参数

#### CART 决策树 (`DecisionTreeRegressor`)
*   **`max_depth` (最大深度)**
    *   **说明**: 树生长的最大层数。
    *   **调整建议**: 默认是不限制。
        *   设置较小值 (如 3, 5): 防止过拟合，模型更简单。
        *   设置较大值: 模型更复杂，可能捕捉更多细节，但也容易过拟合。
*   **`min_samples_split` (分裂所需最小样本数)**
    *   **说明**: 一个节点必须包含多少个样本才能继续分裂。
    *   **调整建议**: 增大该值 (如 5, 10) 可以限制树的生长，防止过拟合。

#### SVR 支持向量回归 (`SVR`)
*   **`kernel` (核函数)**
    *   **说明**: 决定了超平面的形状。
    *   **调整建议**:
        *   `'linear'`: 线性核，适用于线性关系。
        *   `'rbf'`: 径向基核 (默认)，适用于非线性关系。
        *   `'poly'`: 多项式核。
*   **`C` (正则化参数)**
    *   **说明**: 权衡误差容忍度与模型复杂度。
    *   **调整建议**:
        *   `C` 越大 (如 10, 100): 对误差容忍度低，试图拟合所有点，易过拟合。
        *   `C` 越小 (如 0.1, 1): 对误差容忍度高，模型更平滑，易欠拟合。

#### MLP 多层感知机 (`MLPRegressor`)
*   **`hidden_layer_sizes` (隐藏层结构)**
    *   **说明**: 定义神经网络的层数和每层的神经元数量。
    *   **调整建议**:
        *   `(100,)`: 单个隐藏层，100个神经元。
        *   `(100, 50)`: 两个隐藏层。
        *   增加层数或神经元数量通常能增强模型的拟合能力，但也增加了计算量和过拟合风险。
*   **`max_iter` (最大迭代次数)**
    *   **说明**: 训练的最大轮数。
    *   **调整建议**: 如果控制台提示 "ConvergenceWarning"，说明模型未收敛，需要增大此值 (如 2000, 5000)。
*   **`activation` (激活函数)**
    *   **说明**: 决定神经元如何处理输入信号。
    *   **调整建议**: `'relu'` (默认), `'tanh'`, `'logistic'`。不同激活函数适用于不同类型的数据分布。

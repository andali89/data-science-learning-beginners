---
title: 2. 房屋价格预测实施指南
author: Anda Li
date: 2024-03-06
category: Data Science Learning
layout: post
course_previous:
  url: /lesson4/2024-03-05-lesson4-house-price-case/
  title: 1. 房屋价格预测实验要求
course_next:
  url: /lesson4/2026-11-18-lesson4-extensions/
  title: 3. 可视化与回归模型拓展
---

本指南使用 `House Prices Prediction.ipynb` 和 `House Prices Prediction_Advanced_Models.ipynb`。先运行基础 Notebook，再从进阶 Notebook 选用至少一种回归模型。将自己的结果填入实验报告。

<!-- website-only:start -->

<!-- website-only:end -->

## 1. 读取并检查数据

从解压后的 `codes` 目录运行 Notebook。先检查数据规模、缺失值和重复值，再用 `df.info()` 和 `df.describe()` 查看类型和分布。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

```python
df = pd.read_csv('house_price_regression_dataset.csv')
print('数据规模', df.shape)
print('各列缺失值数\n', df.isna().sum())
print('重复记录数', df.duplicated().sum())
df.head()
```

> **Seaborn 版本提示**
>
> 原版 Notebook 已验证可在 Python 3.11、Seaborn 0.10.0、NumPy 1.23.5、pandas 1.5.3、Matplotlib 3.6.3、SciPy 1.10.1 和 scikit-learn 1.2.2 的组合下运行。本次保留了原有的 Seaborn 绘图接口。若系统同时安装了 Seaborn 0.10.0 和 NumPy 1.26.x，热力图和柱状图可能报 `np.bool` 或 `np.float` 错误，先查看当前内核实际加载的版本。
{: .block-warning }

## 2. 探索性数据分析

用直方图和箱线图观察房价分布。箱体覆盖第 25 至第 75 百分位。默认须线延伸到距箱体边缘不超过 1.5 倍四分位距的最远观测值。

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

```python
# 目标变量House_Price分布
plt.figure(figsize = (12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['House_Price'], bins = 50, alpha = 0.7, color = 'skyblue')
plt.xlabel('房屋价格')
plt.ylabel('频数')
plt.title('房屋价格分布')

plt.subplot(1, 2, 2)
sns.boxplot(y = df['House_Price'])
plt.title('房屋价格箱线图')

plt.tight_layout()
plt.show()
```

基础 Notebook 还绘制居住面积与房价的散点图，以及按卧室数量分组的箱线图。用统一的参考年份计算房龄，再绘制 Pearson 相关系数热力图。

```python
df = df.assign(House_Age=2026 - df['Year_Built']).drop(columns='Year_Built')
```

```python
# 相关性热力图
plt.figure(figsize = (10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', center=0, vmin=-1, vmax=1, fmt=".2f")
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()
```

## 3. 数据划分与标准化

> **先划分，再标准化**
>
> 用训练集的均值与标准差进行标准化，测试集复用同一变换。
{: .block-warning }

CART 可以直接使用原尺度特征。

```python
X = df.drop(columns='House_Price')
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 4. 多元线性回归与评价

```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
```

```python
def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(actual, predicted),
    }
```

用 `pd.Series(regression_metrics(y_test, y_pred))` 查看结果。MAE 和 RMSE 使用房价原单位，MSE 使用房价单位的平方。R² 等于 1 表示完美预测，等于 0 表示在这批数据上的平方误差与直接使用真实值均值的基准相同，低于该基准时可为负值。

```python
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([low, high], [low, high], color='gray', linestyle='--')
ax.set(xlabel='真实房价', ylabel='预测房价', title='多元线性回归预测结果')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
plt.show()
```

图中对角线表示预测值等于真实值。点在对角线上方表示高估，在下方表示低估。

## 5. 比较备选模型

进阶 Notebook 用同一划分运行线性回归、CART、线性 SVR 和 MLP，汇总同一测试集上的评价指标。完成实验时，从后三种模型中至少选一种。使用实际运行的模型名称和参数，比较同一指标。可参阅「可视化与回归模型拓展」中的示例。

## 6. 解读回归系数

```python
coefficients = pd.Series(lr_model.coef_, index=X.columns, name='回归系数')
order = coefficients.abs().sort_values().index
fig, ax = plt.subplots(figsize=(8, 4))
coefficients.loc[order].plot(kind='barh', ax=ax)
ax.axvline(0, color='gray', linewidth=1)
ax.set_xlabel('回归系数（每增加一个训练集标准差）')
fig.tight_layout()
plt.show()
```

系数表示在控制其他入模特征时，该特征每增加一个训练集标准差对应的预测房价变化。绝对值便于比较模型中的关联强度，正负号给出方向。相关性较强的特征可能让系数对样本变化较敏感。

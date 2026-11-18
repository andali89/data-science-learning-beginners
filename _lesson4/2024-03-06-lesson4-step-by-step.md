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

本指南使用 `House Prices Prediction.ipynb` 和 `House Prices Prediction_Advanced_Models.ipynb`。先检查数据、绘制图表并拟合多元线性回归模型，再从进阶 Notebook 中至少选择一种回归模型进行比较。按顺序运行下面的代码，每运行一段就核对结果。根据自己的运行记录和分析撰写报告。

## 1. 读取并检查数据

### 1.1 工作目录与导入

解压后，将两个 Notebook 与 `house_price_regression_dataset.csv` 放在同一个 `codes` 目录。代码使用相对路径，Python 会在内核的当前工作目录下查找该文件。`Path.cwd()` 返回当前工作目录，`is_file()` 检查指定路径是否指向文件。

```python
from pathlib import Path

data_path = Path('house_price_regression_dataset.csv')
print('当前工作目录', Path.cwd())
print('找到 CSV', data_path.is_file())
```

**核对 CSV 路径**

第二行应显示 `True`。如果显示 `False`，先核对工作目录和文件名。Notebook 所在目录和内核工作目录可能不同，以 `Path.cwd()` 的输出为准。

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

`import pandas as pd` 给 pandas 起了别名，后面用 `pd.read_csv()` 就能调用它的函数。`np` 用于数值运算，`plt` 和 `sns` 用于绘图。`from ... import ...` 直接导入指定的函数或类，所以可以直接写 `LinearRegression()`。括号中的值是传给函数的参数，像 `bins=50` 这样写的是带名称的参数。

> **Seaborn 版本提示**
>
> 原版 Notebook 曾在 Python 3.11、Seaborn 0.10.0、NumPy 1.23.5、pandas 1.5.3、Matplotlib 3.6.3、SciPy 1.10.1 和 scikit-learn 1.2.2 的环境中成功运行。下面沿用原版的 Seaborn 绘图接口。若同时使用 Seaborn 0.10.0 和 NumPy 1.26.x 时出现 `np.bool` 或 `np.float` 错误，可先打印 `sns.__version__` 和 `np.__version__`，核对当前内核加载的版本。
{: .block-warning }

### 1.2 表格、字段与数据质量

`pd.read_csv(data_path)` 读取逗号分隔文件，返回一个 `DataFrame`。它像一张带列名的表，每行是一条房屋记录。`df['House_Price']` 取出一列，得到带行索引的 `Series`。选取多列时，如 `df[['Square_Footage', 'House_Price']]`，得到的仍是 `DataFrame`。

```python
df = pd.read_csv(data_path)
print('数据规模', df.shape)
print(df.head(3))
df.info()
print('各列缺失值数\n', df.isna().sum())
print('重复记录数', df.duplicated().sum())
print(df.describe())
```

| 写法 | 做什么 | 如何看结果 |
| --- | --- | --- |
| `df.shape` | 返回 `(行数, 列数)` | 这是属性，直接读取。本文件为 `(1000, 8)`。 |
| `df.head(3)` | 返回最前面的三行。省略 `3` 时默认取五行 | 用来核对列名、样本格式和读入内容。 |
| `df.info()` | 打印列类型与非空数，返回 `None` | 直接调用即可。非空数少于总行数的列包含缺失。 |
| `df.isna().sum()` | `isna()` 生成逐格布尔表，`sum()` 默认按列汇总 | `True` 按 1 累加，因此结果是每列的缺失数。 |
| `df.duplicated().sum()` | `duplicated()` 返回每行是否与前面的整行重复，默认保留首次出现 | 求和得到首次出现之外的重复记录数。 |
| `df.describe()` | 返回数值列的描述统计表 | 查看 `count`、`mean`、`std`、四分位数和极值。 |

**数据检查结果**

本实验的 CSV 有 1,000 行、8 列。各列缺失值数均为 0，完全重复的记录数也为 0。`House_Price` 是预测目标，其余七列用于构造特征。各字段的含义见实验要求。例如，卧室数以整数存储，表示每套房屋的卧室数量。

**自己动手**

用 `describe()` 找到房价中位数，并说明统计表中哪一行对应中位数。报告只需记录与实验有关的检查结果。

## 2. 探索性数据分析

### 2.1 用直方图和箱线图观察房价

直方图显示房价集中在哪些区间，箱线图显示中间一半样本的房价范围，并标出距离箱体较远的观测值。把两张图放在同一画布上，便于结合频数和分位数解读房价分布。

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['House_Price'], bins=50, alpha=0.7, color='skyblue')
plt.xlabel('房屋价格')
plt.ylabel('频数')
plt.title('房屋价格分布')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['House_Price'])
plt.title('房屋价格箱线图')

plt.tight_layout()
plt.show()
```

- `rcParams` 是 Matplotlib 的绘图设置。字体列表提供候选字体，具体显示取决于本机已安装的字体。`axes.unicode_minus=False` 使用普通连字符显示负号。
- `plt.figure(figsize=(12, 4))` 创建一个 `Figure`，尺寸按英寸计。`plt.subplot(1, 2, 1)` 在一行两列的布局中选定第一块绘图区，返回一个 `Axes`。后续 `plt.xlabel()` 和 `plt.title()` 设置当前绘图区的标签和标题。
- `plt.hist(x, bins=50)` 将数值分成 50 个区间，默认柱高是该区间内的观测数。它返回频数、区间边界和绘制的柱形对象。区间太少容易掩盖局部变化，区间太多则容易让少量观测造成的波动显得明显。
- `sns.boxplot(y=...)` 画纵向箱线图并返回 `Axes`。箱体覆盖第 25 至第 75 百分位，中线是中位数。默认须线延伸到不超过箱体边缘 1.5 倍四分位距的最远观测值。须线外的点值得核查，保留或删除应根据记录本身的情况决定。
- `plt.tight_layout()` 调整元素间距，`plt.show()` 显示图表。

**自己动手**

将 `bins` 改为 20，比较修改前后的直方图。写出两张直方图共同支持的一项观察，并注明报告采用的分箱数。

### 2.2 用散点图观察面积与售价

`plt.scatter(x, y, alpha=0.5)` 按相同位置配对两列数值，每行画一个点，返回可进一步设置样式的散点对象。`alpha` 控制透明度，点重叠时更容易看出密集区域。

```python
plt.figure(figsize=(7, 4))
plt.scatter(df['Square_Footage'], df['House_Price'], alpha=0.5)
plt.xlabel('居住面积（平方英尺）')
plt.ylabel('房屋售价')
plt.tight_layout()
plt.show()
```

**散点图怎么看**

横轴为面积，纵轴为售价。先描述点云的方向和形状，再找出偏离整体趋势较大的点。分析卧室数等离散变量时，可参考拓展页的分组箱线图，比较各组的房价分布。

### 2.3 构造房龄并读取相关性热力图

以 2026 年为参考年，用 `2026 - df['Year_Built']` 逐行计算房龄。`assign(House_Age=...)` 返回包含房龄列的表，`drop(columns='Year_Built')` 返回删去建造年份列的表。房龄与建造年份可以互相换算，保留其中一个即可。这里将建模数据表保存为 `model_df`，原始 `df` 仍可用于数据检查和绘图。下载的 Notebook 将建模数据表记作 `df`，两种写法保留的建模字段相同。

```python
model_df = df.assign(House_Age=2026 - df['Year_Built']).drop(columns='Year_Built')
print(model_df[['House_Age', 'House_Price']].head(3))

correlation_matrix = model_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, annot=True, cmap='coolwarm',
    center=0, vmin=-1, vmax=1, fmt='.2f',
)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()
```

`corr()` 默认计算 Pearson 相关系数，返回行列都按字段命名的对称矩阵。对角线表示变量与自身的相关性。在 `House_Price` 一行或一列查找其他字段，可以看到它们与售价的线性关联方向和大小。`heatmap()` 返回 `Axes`，`annot=True` 在格子中显示相关系数，`fmt='.2f'` 保留两位小数，`vmin=-1` 和 `vmax=1` 固定色标范围。`center=0` 将零设为颜色分界，`coolwarm` 用冷色和暖色区别负值与正值。

**相关系数怎么看**

相关系数为正时，两个变量倾向于同向变化。系数接近 0 时，两个变量的线性关联较弱，仍可能存在非线性关系。这里对既定字段做描述性分析。如果需要根据数据挑选特征或设计模型，用训练集作判断，把测试集留给模型评价。

## 3. 数据划分与标准化

### 3.1 分离特征和目标，再划分样本

`X` 的每一行包含一套房屋的特征，`y` 中对应的值是这套房屋的实际售价。从 `X` 中移除 `House_Price`，使输入符合预测时只有房屋特征可用的情形。将两个对象同时传给 `train_test_split(X, y, ...)`，即可按相同的行位置划分特征和目标。

```python
X = model_df.drop(columns='House_Price')
y = model_df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
print('特征的训练与测试形状', X_train.shape, X_test.shape)
print('房价的训练与测试形状', y_train.shape, y_test.shape)
```

`test_size=0.2` 将 20% 的记录留作测试。函数依次返回训练特征、测试特征、训练目标和测试目标，赋值语句左侧的四个变量按同一顺序接收这些结果。`X_train` 与 `y_train` 的记录一一对应，测试特征与测试目标也是如此。

**核对训练集和测试集的形状**

特征部分的形状分别为 `(800, 7)` 和 `(200, 7)`，目标部分分别是 `(800,)` 和 `(200,)`。比较模型时复用这些划分，避免因测试样本不同而造成分数差异。

### 3.2 只用训练集学习标准化参数

面积和房龄的数值尺度不同。标准化时，每个特征值先减去该特征在训练集中的均值，再除以训练集中的标准差。转换后的值表示它离训练均值有多少个标准差，正负号表示偏离方向。

> **先划分，再标准化**
>
> `fit_transform(X_train)` 先学习训练特征的统计量，再转换训练特征。`transform(X_test)` 只复用这些统计量，避免将测试信息带入训练。CART 按特征阈值划分，可直接使用原尺度特征。
{: .block-warning }

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('标准化后的形状', X_train_scaled.shape, X_test_scaled.shape)
```

`StandardScaler()` 创建一个尚未拟合的对象。`fit()` 学习统计量并返回该对象，`transform()` 返回转换后的数值数组，`fit_transform()` 将两步合在一起。标准化改变数值尺度，保留行列顺序和形状。在这个例子中，数组不带 pandas 的列名，因此保留原来的 `X.columns` 来解释模型系数。`y_train` 与 `y_test` 仍使用房价原尺度。

**核对标准化后的形状**

两份数组仍为 `(800, 7)` 和 `(200, 7)`。用 `fit_transform(X_test)` 会让测试集使用另一套标准，不能和训练时的输入尺度直接对应。

## 4. 多元线性回归与评价

### 4.1 拟合模型并得到预测

多元线性回归把房价表示为截距加上各特征的加权和。`LinearRegression()` 建立普通最小二乘模型。拟合时为七个特征学习权重，使训练集上预测误差的平方和尽可能小。

```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
print('预测形状', y_pred.shape)
```

`fit(X, y)` 学习模型并返回拟合后的模型对象。`predict(X)` 应用已学到的参数，不需要传入真实房价。这里返回一维数组 `y_pred`，其中第一个值对应 `y_test` 的第一个真实价格。输入列数、列顺序和尺度都应与训练时一致。

### 4.2 理解四个评价指标

先用三个便于手算的值理解误差。真实值为 `100, 200, 300`，预测值为 `110, 180, 330`。三项绝对误差是 10、20、30，平方误差是 100、400、900。各个指标对这些误差的处理不同。

| 指标与调用 | 如何计算 | 如何读出结果 |
| --- | --- | --- |
| `mean_absolute_error(actual, predicted)` | 平均绝对误差（MAE），例中为 20 | 平均偏差的大小，单位与房价相同。越低越好。 |
| `mean_squared_error(actual, predicted)` | 平均平方误差（MSE），例中约为 466.67 | 大误差会被平方放大，单位为房价单位的平方。越低越好。 |
| `np.sqrt(mse)` | 对 MSE 开平方得到 RMSE，例中约为 21.60 | 回到房价单位，仍对大误差较敏感。越低越好。 |
| `r2_score(actual, predicted)` | 用预测误差平方和与均值基准的误差平方和比较，例中为 0.93 | 1 表示完美预测，0 与该组真实值均值的平方误差基准相当，更差时可为负值。 |

下面用 `def` 把相同的评分步骤写成一个函数。`actual` 和 `predicted` 是参数，`return` 输出一个按名称存储指标的字典。定义函数后还要调用它才会计算结果。每次都传入等长、逐行对应的实际值和预测值，参数顺序尤其会影响 R² 的解释。

```python
def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(actual, predicted),
    }

example_scores = regression_metrics([100, 200, 300], [110, 180, 330])
print(pd.Series(example_scores).round(2))

scores = regression_metrics(y_test, y_pred)
print(pd.Series(scores))
```

`pd.Series(scores)` 将字典转为带标签的一维数据，字典的键用作索引。`round(2)` 将结果四舍五入到两位小数，便于核对手算结果。第二组输出是本实验的测试结果，报告应解释实际运行得到的指标。R² 衡量模型相对于均值基准的表现，不能直接解读为预测准确率的百分比。

### 4.3 画真实值与预测值对照图

`plt.subplots(figsize=(6, 6))` 同时返回画布 `fig` 和绘图区 `ax`。在 `ax` 上绘制散点和参考线，并设置坐标轴标签。参考线的横坐标和纵坐标都使用 `[low, high]`，因此两个端点各自的横纵坐标相等。

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

`min()` 和 `max()` 取真实值与预测值共同的范围。`ax.plot()` 返回绘制的线对象列表，`ax.set_aspect('equal')` 让两轴每个单位在图上等长。点在对角线上方表示高估，在下方表示低估。先看点云与对角线的总体距离，再找是否某个价格区间的偏差更集中。

## 5. 比较备选模型

进阶 Notebook 使用相同的训练集和测试集运行线性回归、CART、线性 SVR 和 MLP。完成实验时，从后三种模型中至少选一种，与多元线性回归比较。

把 `X_train`、`X_test` 及其标准化版本留在内核中。拓展页说明每种模型选用哪组特征，并使用上面的 `regression_metrics(y_test, prediction)` 评分。报告记录模型名称、实际参数和共同指标，解释同一测试集上的差异。有调参需要时，先在训练集内验证。

## 6. 解读回归系数

拟合后的 `lr_model.coef_` 按训练特征的顺序存放系数。将系数和 `X.columns` 配对后，用绝对值确定展示顺序，再用原有符号画图。

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

`pd.Series(..., index=X.columns)` 为每个系数赋予字段名。`abs()` 返回绝对值，`sort_values()` 将大小从小到大排序。`.index` 取出排序后的字段名，`.loc[order]` 再按这些名字从原始系数中取值，保留正负号。`kind='barh'` 画横向柱状图，零线右边是正系数，左边是负系数。

例如，某系数若为 20,000，表示在其他入模特征固定时，该特征每增加一个训练集标准差，对应的预测房价增加约 20,000 个价格单位。特征之间相关性较强时，系数和排名可能随样本变化。报告用它解释模型中的关联。要判断装修或扩建本身对售价的影响，还需要其他证据。

### 报告检查与排查

**报告内容清单**

记录数据检查结果，提供带说明的探索性分析图表、模型设置和基于同一测试集的指标对比表。指出系数图中的一项发现，并说明它如何帮助理解模型预测及其局限。

| 现象 | 优先检查 |
| --- | --- |
| `FileNotFoundError` | 核对 `Path.cwd()`、CSV 文件名和解压位置。 |
| `NameError` | 重启内核后从上到下运行。修改上游变量后，重跑用到它的代码。 |
| 输入维度不一致 | 检查 `House_Price` 是否误入 `X`，以及训练与测试特征的列顺序和形状。 |
| 中文显示方框 | 核对机器实际安装的中文字体及 `font.sans-serif` 设置。 |

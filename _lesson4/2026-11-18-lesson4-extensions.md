---
title: 3. 可视化与回归模型拓展
author: Anda Li
date: 2026-11-18
category: Data Science Learning
layout: post
course_previous:
  url: /lesson4/2024-03-06-lesson4-step-by-step/
  title: 2. 房屋价格预测实施指南
---

本页为选读。先完成实施指南的第 1 至第 4 节，再按需要选择图表或模型。实验要求在多元线性回归之外至少运行一种备选模型。每种备选模型的代码都可在完成基准建模后单独运行。

## 图表选择

先明确要回答的问题，再选择图表。标明图中的变量、单位和分组依据，使读者仅凭图表就能理解这些信息。

| 要回答的问题 | 用什么图 | 优先看什么 |
| --- | --- | --- |
| 房价集中在哪些区间？ | 直方图 | 频数、分箱数和分布形状。 |
| 卧室数不同的房屋，房价分布有何差异？ | 分组箱线图 | 各组中位数、四分位距和组内样本量。 |
| 面积与房价呈现怎样的关系？ | 散点图 | 趋势、离散程度和偏离趋势的点。 |
| 哪些字段之间有较强的线性关联？ | 相关性热力图 | 用统一的 −1 至 1 色标比较相关系数。 |
| 模型在哪些价位容易高估或低估房价？ | 真实值与预测值对照图，或残差图 | 对照图显示预测值与真实值的差距，残差图直接呈现误差随预测价位的变化。 |

### 分组箱线图与样本量

在原始数据表 `df` 中，先用 `groupby('Num_Bedrooms')` 按卧室数分组，再选出每组的 `House_Price`。`agg(['count', 'median'])` 统计各组非缺失售价的数量，并计算中位数，返回以卧室数为索引的表。箱线图展示各组的房价分布，汇总表列出各组的样本量和中位数。

```python
bedroom_summary = df.groupby('Num_Bedrooms')['House_Price'].agg(['count', 'median'])
print(bedroom_summary)

fig, ax = plt.subplots(figsize=(8, 4))
df.boxplot(column='House_Price', by='Num_Bedrooms', grid=False, ax=ax)
ax.set(xlabel='卧室数量', ylabel='房屋售价', title='按卧室数量分组的房价')
fig.suptitle('')
fig.tight_layout()
plt.show()
```

`column` 指定各组要画的数值列，`by` 指定分组列，`ax` 让箱线图落在已经创建的绘图区。`fig.suptitle('')` 清除 pandas 自动生成的整张图标题，保留 `ax.set()` 设置的标题。

**分组箱线图怎么看**

按卧室数将样本分为 1 至 5 间卧室的五组，每个箱体对应一组。若某组的中位数较高，先看各组的样本量和分布重叠程度，再考虑该组房屋的面积等特征是否也有差异。这张图展示组间差异，单凭它无法估计增加一间卧室对房价的影响。

### 残差图

运行实施指南中的代码得到 `y_test` 和 `y_pred` 后，即可计算每套房屋的残差。残差是**真实值减预测值**。例如，实际售价为 300,000，预测售价为 280,000，残差为 20,000，表示模型低估了售价。实施指南中的对照图用对角线上方的点表示高估。本节残差图用零线上方的点表示低估。

```python
residuals = y_test.to_numpy() - y_pred
fig, ax = plt.subplots()
ax.scatter(y_pred, residuals, alpha=0.6)
ax.axhline(0, color='gray', linestyle='--')
ax.set(xlabel='预测房价', ylabel='真实房价减预测房价')
fig.tight_layout()
plt.show()
```

`y_test.to_numpy()` 返回真实值的一维数组。两个数组按位置相减，因此要保留生成预测时的测试行顺序，单独排序任一数组都会配错房屋。`ax.axhline(0)` 添加横跨绘图区的零线，返回绘制的线对象。每个点的横坐标是预测房价，纵坐标是相应的残差。

- 点大体围绕零线，说明从图上未见贯穿所有价位的单向偏差。
- 某个价格区间的点多在零线上方或下方，提示模型可能在该区间系统性低估或高估。
- 如果点云随预测价位升高而变宽，较贵房屋的绝对误差可能更分散。若残差呈明显弧形，可以在训练集验证遗漏的非线性关系。

**自己动手**

挑一个价格区间，描述点相对于零线的位置和分散程度。用图上的证据写结论。如果想据此调整模型，回到训练集验证新方案。

## 备选模型

### 准备输入与记录基准

运行本节示例前，先完成实施指南中的基准模型。若已重启内核，需先重新运行实施指南中的代码，生成下列对象。

| 对象 | 本页用途 |
| --- | --- |
| `X_train`、`X_test` | 原尺度特征，供 CART 和交叉验证示例使用。 |
| `X_train_scaled`、`X_test_scaled` | 用训练特征拟合后得到的标准化数组，供线性 SVR 和 MLP 使用。 |
| `y_train`、`y_test` | 原尺度的实际房价，用于训练或计算评价指标。 |
| `y_pred`、`regression_metrics()` | 多元线性回归的测试预测，以及实施指南定义的评分函数。 |

先导入估计器，并将基准模型的预测存入 `predictions` 字典。这个字典以模型名称为键，以对应的测试集预测为值。接下来可运行任意一种备选模型，也可依次运行三种模型。每段代码都会将生成的预测存入字典。

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

predictions = {'OLS': y_pred}
```

### CART 决策树回归

CART 通过特征阈值将房屋记录分组。例如，它可能按居住面积划分样本，再在组内按其他特征细分。在本例使用的默认平方误差准则下，落入同一个叶节点的房屋得到该叶节点训练房价的均值作为预测。可以用 CART 查看阈值式的分组能否补充线性基准。

```python
cart_model = DecisionTreeRegressor(random_state=123)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
predictions['CART'] = y_pred_cart
```

- `max_depth` 设置树的最大深度，默认值为 `None`，表示不设深度上限。`min_samples_leaf` 指定每个叶节点至少需要的训练样本数。较浅的树或样本较多的叶节点通常对应更粗略的分组规则。
- `fit(X_train, y_train)` 使用原尺度特征和售价学习划分，返回拟合后的树。`predict(X_test)` 将每行测试特征送入树，返回一维的房价预测数组。按阈值比较特征时可以直接使用原尺度输入。

**自己动手**

观察 CART 与线性回归的测试误差差异。若想比较不同深度，使用页末的训练集验证示例。

### 线性 SVR

在线性支持向量回归（SVR）中，`epsilon` 设定拟合时可容许的误差范围。绝对误差在这个范围内时，不计入拟合损失。`C` 调节范围外误差在训练中的权重。本例保留售价的原始尺度，因此 `epsilon` 与房价的单位相同。

```python
svr_model = SVR(kernel='linear', C=1000)
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
predictions['SVR'] = y_pred_svr
```

- `kernel='linear'` 使用线性核，适合与普通最小二乘比较两种线性方法怎样处理误差。`C=1000` 是本示例的配置，较大的 `C` 会更重视减小带外的训练误差，是否有益要用验证数据判断。
- `fit()` 在标准化特征上学习并返回模型。`predict()` 接收用同一特征缩放器转换的测试数据，返回原尺度的一维售价数组。即使两个特征矩阵形状相同，将原尺度的 `X_test` 传给用标准化特征训练的模型也会改变预测含义。
- 若想做非线性对照，可以把 `kernel='rbf'` 作为验证阶段的候选配置。先固定比较条件，再评估是否有改善。

### MLP

MLP 通过隐藏层学习特征组合，可以表达比线性回归更灵活的关系。这里分别标准化特征和训练集中的房价，使模型能在较合适的数值尺度上优化。特征沿用实施指南中拟合的标准化规则，训练房价使用单独的 `target_scaler`。

```python
target_scaler = StandardScaler()
y_train_column = y_train.to_numpy().reshape(-1, 1)
y_train_scaled = target_scaler.fit_transform(y_train_column).ravel()

mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123)
mlp_model.fit(X_train_scaled, y_train_scaled)
y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
y_pred_mlp = target_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()
predictions['MLP'] = y_pred_mlp
```

- `reshape(-1, 1)` 把 `(800,)` 的训练售价数组变成 `(800, 1)`。`-1` 让 NumPy 根据元素总数推算行数。缩放器接收二维表，`ravel()` 再将这一列展平为 MLP 所需的一维目标 `(800,)`。
- `hidden_layer_sizes=(100, 50)` 表示两层隐藏层，分别有 100 和 50 个单元。`max_iter=1000` 是本例的迭代上限。若出现 `ConvergenceWarning`，先检查缩放步骤和数据有无异常，再考虑在训练集上验证其他配置。
- `predict(X_test_scaled)` 返回 200 个**标准化后的预测值**。`inverse_transform()` 用训练目标的缩放规则恢复房价单位，返回二维数组，再以 `ravel()` 得到形状 `(200,)` 的 `y_pred_mlp`。此时才能和原尺度的 `y_test` 比较。

**确认预测值已恢复到房价尺度**

房价缩放器只接收 `y_train`，度量误差时传入 `y_pred_mlp`。若预测值大多接近 0 或 1，而实际售价仍是原尺度，先检查是否漏了逆变换。

### 同一测试集上的比较

运行基准与至少一种备选模型后，运行以下比较代码。`predictions.items()` 提供每个模型名称和它的预测数组。循环内调用同一个评分函数，用 `row.update()` 把各项指标写入当前行字典。这个方法直接修改 `row`，返回 `None`。`append()` 把每行加入列表，`pd.DataFrame(rows)` 再将行列表转成表。

```python
rows = []
for name, prediction in predictions.items():
    row = {'Model': name}
    row.update(regression_metrics(y_test, prediction))
    rows.append(row)

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Model', y='R2', data=results_df, ax=ax)
ax.axhline(0, color='gray', linewidth=1)
ax.set(ylabel=r'$R^2$', title='同一测试集上的模型表现')
fig.tight_layout()
plt.show()
```

`to_string(index=False)` 返回便于打印的文本表并省略行索引。`sns.barplot()` 从 `data` 指定的表中读取 `x` 和 `y`，返回 `Axes`。这里只给每个模型一条 R² 记录，柱高就是该模型的测试分数。保留零线和自动纵轴范围，负 R² 也能显示。

**核对模型比较表**

只运行一种备选模型时，结果表应有两行。运行三种时应有四行。每行都应使用同一组 200 个测试房价。若表里只有 OLS，先检查是否运行了所选模型的登记行。重新执行 `predictions = {'OLS': y_pred}` 会清空已登记的备选模型，需要重新运行相应模型块。

## 选做参数实验

在其他条件相同的情况下，比较不同的 `max_depth` 设置，找出在训练集内验证误差更小的候选深度。

> **在训练集内调参**
>
> 用 `X_train` 和 `y_train` 选择参数。确定方案后，再用保留的 `X_test` 和 `y_test` 做最终评价。
{: .block-warning }

用 `KFold` 将 800 条训练记录随机划分为五折。每轮用其中四折拟合模型，再用剩下的一折验证。`GridSearchCV` 按 `param_grid` 遍历三个候选深度，使用相同的五折划分比较各候选项。

```python
from sklearn.model_selection import GridSearchCV, KFold

folds = KFold(n_splits=5, shuffle=True, random_state=123)
search = GridSearchCV(
    DecisionTreeRegressor(random_state=123),
    param_grid={'max_depth': [3, 5, None]},
    scoring='neg_mean_absolute_error',
    cv=folds,
)
search.fit(X_train, y_train)
print('选中的参数', search.best_params_)
print('最佳平均验证 MAE', -search.best_score_)

y_pred_cart_tuned = search.predict(X_test)
print('最终测试指标', regression_metrics(y_test, y_pred_cart_tuned))
```

- `scoring='neg_mean_absolute_error'` 用负的 MAE 作为评分，因为搜索工具始终选择分数较大的设置。`best_score_` 是选中设置的平均验证分数，加负号得到通常报告的 MAE。
- `best_params_` 返回选中的配置字典。默认 `refit=True` 会让搜索对象用该配置在全部训练数据上重新拟合。之后 `search.predict(X_test)` 使用这一模型返回最终测试预测。
- 这里的 CART 可以直接使用原尺度特征。如果验证的是需要标准化的模型，可用管道把标准化和模型放在一起，使每一折用各自的训练部分拟合缩放器。先标准化整个外层训练集再交叉验证，会让内部验证折的信息进入缩放规则。

**自己动手**

记录候选深度、验证指标和最终采用的配置。在报告中区分平均验证 MAE 和最终测试集的结果。

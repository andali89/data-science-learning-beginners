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

本页为选读。实验要求比较多元线性回归与至少一种备选模型。下面的画图和调参示例可用于进一步分析。

<!-- website-only:start -->
[实验要求与下载]({{ site.baseurl }}/lesson4/2024-03-05-lesson4-house-price-case/) · [实施指南]({{ site.baseurl }}/lesson4/2024-03-06-lesson4-step-by-step/)
<!-- website-only:end -->

## 图表选择

| 要回答的问题 | 图表与读图方法 |
| --- | --- |
| 房价集中在哪些区间？ | 直方图。注明分箱设置与单位，观察分布形状。 |
| 不同卧室数量的房价分布是否相似？ | 分组箱线图。比较各组中位数与四分位距，并检查组内样本量。 |
| 面积与房价的关系是什么形状？ | 散点图。看趋势、离散程度及异常点。 |
| 哪些变量之间有较强的线性关联？ | 相关性热力图。用统一的 −1 至 1 色标，查看相关系数。 |
| 模型在哪些价位容易偏高或偏低？ | 真实值与预测值对照图，或以预测值为横轴的残差图。 |

### 残差图

在基础 Notebook 得到 `y_test` 和 `y_pred` 后，可绘制残差。残差等于真实值减预测值，正值表示低估，负值表示高估。查看点是否围绕零线、误差是否随价位变化。

```python
residuals = y_test.to_numpy() - y_pred
fig, ax = plt.subplots()
ax.scatter(y_pred, residuals, alpha=0.6)
ax.axhline(0, color='gray', linestyle='--')
ax.set(xlabel='预测房价', ylabel='真实房价减预测房价')
fig.tight_layout()
plt.show()
```

## 备选模型

以下代码接续实施指南中的数据划分。进阶 Notebook 提供同样的示例。以下导入和模型代码可按需要运行。

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
```

### CART 决策树回归

CART 按特征阈值分割样本。在测试误差较大时，可在训练集内尝试限制 `max_depth` 或提高最小叶节点样本数。

```python
cart_model = DecisionTreeRegressor(random_state=123)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
```

### 线性 SVR

示例使用 `kernel='linear'` 和 `C=1000`。若选做非线性对照，可在训练集内验证 RBF 核的配置。

```python
svr_model = SVR(kernel='linear', C=1000)
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
```

### MLP

MLP 训练时采用标准化特征和训练房价。预测后用训练目标的标准化规则逆变换，再以房价原尺度比较各模型。

```python
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123)
mlp_model.fit(X_train_scaled, y_train_scaled)
y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
y_pred_mlp = target_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()
```

### 同一测试集上的比较

运行了四个模型时，可使用以下代码。只运行部分备选模型时，在 `predictions` 中填写线性回归和实际运行的模型。

```python
predictions = {
    'OLS': y_pred,
    'CART': y_pred_cart,
    'SVR': y_pred_svr,
    'MLP': y_pred_mlp,
}
results_df = pd.DataFrame([
    {'Model': name, **regression_metrics(y_test, prediction)}
    for name, prediction in predictions.items()
])
print(results_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Model', y='R2', data=results_df, ax=ax)
ax.axhline(0, color='gray', linewidth=1)
ax.set(ylabel=r'$R^2$', title='同一测试集上的模型表现')
fig.tight_layout()
plt.show()
```

## 选做参数实验

> **在训练集内调参**
>
> 以下示例仅在训练集内部用 5 折交叉验证比较 CART 的深度。按平均 MAE 选择深度，用完整训练集重新训练选定模型，最后用测试集评估。
{: .block-warning }

若给需要标准化的模型调参，可将标准化放在管道内，每折各自拟合。

```python
from sklearn.model_selection import GridSearchCV, KFold
folds = KFold(n_splits=5, shuffle=True, random_state=123)
search = GridSearchCV(
    DecisionTreeRegressor(random_state=123),
    {'max_depth': [3, 5, None]},
    scoring='neg_mean_absolute_error', cv=folds,
)
search.fit(X_train, y_train)
print(search.best_params_)
print(regression_metrics(y_test, search.predict(X_test)))
```

---
title: 3. 备选分类模型使用方法
author: Anda Li
date: 2026-11-11
category: Data Science Learning
layout: post
course_previous:
  url: /lesson3/2026-11-11-02-lesson3-guide/
  title: 2. 企业员工流失预警实施与调参指南
course_next:
  url: /lesson4/2024-03-05-lesson4-house-price-case/
  title: 1. 房屋价格预测实验教学案例
---

本页供你选择第三种分类器，不要求把所有模型都运行一遍。先完成[实施与调参指南]({{ site.baseurl }}/lesson3/2026-11-11-02-lesson3-guide/)的划分与预处理，再阅读所选模型。示例仅展示构造、训练和预测；评价图表、参数实验与最终结论需自行完成。

所有示例使用相同的 `X_train_ready`、`X_test_ready` 和 `y_train`。验证调参时，应改用主指南中重新预处理的内部训练/验证数据，不能反复用测试集选参数。模型差异不应与数据划分差异混淆。

## 1. 决策树：容易展示分类规则

决策树通过连续的条件划分做预测，适合把规则解释给管理人员。树过深可能记住训练样本，因此先从限制深度、增加叶节点最少样本数开始。

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=50,
    class_weight="balanced", random_state=42
)
dt.fit(X_train_ready, y_train)  # 学习划分规则。
dt_pred = dt.predict(X_test_ready)  # 返回 0/1 类别，不是概率。
```

`max_depth` 控制最大深度，`min_samples_leaf` 与 `min_samples_split` 限制继续生长；`criterion` 可比较 `"gini"`、`"entropy"`；`max_features` 控制每次分裂考虑的字段数。`class_weight` 调整损失权重，不改变样本数。

可以使用 `plot_tree` 或 `export_text` 展示规则，`feature_names` 必须使用编码后的列名；画整棵树可能过于拥挤，可限制展示深度。树重要性没有方向。**自己动手**：固定其他参数，比较不同深度下的训练和验证表现。

## 2. RBF 支持向量机：学习非线性边界

SVC 利用核函数处理非线性关系，对尺度敏感，所以应沿用主指南的数值标准化。

```python
from sklearn.svm import SVC

svc = SVC(kernel="rbf", C=1.0, gamma="scale",
          class_weight="balanced", probability=False, random_state=42)
svc.fit(X_train_ready, y_train)
svc_pred = svc.predict(X_test_ready)
svc_score = svc.decision_function(X_test_ready)  # 连续分数，可用于 ROC。
```

`C` 越小正则约束越强；RBF 的 `gamma` 越大，单个样本的影响范围越局部。两者共同影响边界，效果需验证。其他核包括 `"linear"`、`"poly"`、`"sigmoid"`，不必全部尝试。

原指南使用 `probability=True`；这里默认关闭额外概率估计以减少训练开销，ROC 直接用决策分数即可。如果研究概率阈值，须在拟合前启用概率输出，并检查 `classes_`。**自己动手**：选 `C` 或 `gamma` 比较验证指标，不把分数当作离职概率。

## 3. LinearSVC：更快的线性对照

LinearSVC 可用于线性边界和较高维输入。它与 SVC 的线性核实现不同，不能因为都叫 SVM 就忽略配置差异。

```python
from sklearn.svm import LinearSVC

linear_svc = LinearSVC(C=1.0, class_weight="balanced",
                       max_iter=5000, random_state=42)
linear_svc.fit(X_train_ready, y_train)
linear_pred = linear_svc.predict(X_test_ready)
linear_score = linear_svc.decision_function(X_test_ready)
```

`C` 控制正则约束；`max_iter` 是迭代上限。若尝试 L1 正则，使用兼容的 `penalty="l1", dual=False` 配置。ROC 可直接用决策分数；确实需要概率时再研究 `CalibratedClassifierCV(method="sigmoid")`。**自己动手**：与逻辑回归比较离职类 Recall/F1，解释线性模型的局限。

## 4. K 最近邻：根据相似员工投票

KNN 根据附近样本预测类别，距离会受量纲影响，必须缩放数值特征。训练通常简单，但预测时计算邻居可能较慢。

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, weights="distance",
                           metric="minkowski", p=2)
knn.fit(X_train_ready, y_train)
knn_pred = knn.predict(X_test_ready)
```

`n_neighbors` 越大，预测通常越平滑；`weights="distance"` 让近邻权重更高，不保证解决类别不平衡。Minkowski 距离中 `p=1` 为曼哈顿距离，`p=2` 为欧氏距离；也可研究 `cosine`。`algorithm` 如 `"auto"`、`"kd_tree"` 影响搜索方式和速度，并非必然改善效果。**自己动手**：比较不同 k 的验证指标，必要时记录预测耗时。

## 5. Gradient Boosting：逐步改进弱学习器

梯度提升顺序建立多棵树以改进预测，与随机森林的建模方式不同。更复杂不一定更好，应关注验证表现。

```python
from sklearn.ensemble import GradientBoostingClassifier

gboost = GradientBoostingClassifier(
    learning_rate=0.05, n_estimators=200, max_depth=3, random_state=42
)
gboost.fit(X_train_ready, y_train)
gboost_pred = gboost.predict(X_test_ready)
```

`learning_rate` 控制每一步的贡献，`n_estimators` 控制树数量，需要联合考虑，没有固定反比或乘积规律。`max_depth`、`max_leaf_nodes`、`min_samples_split`、`min_samples_leaf` 控制弱学习器复杂度；`subsample<1.0` 引入行抽样，可能缓解过拟合。**自己动手**：先固定树数量比较学习率，再设计独立实验研究树数量，记录验证表现。

XGBoost、LightGBM 可作为课外扩展，需要额外安装并学习各自 API；本实验不依赖它们。

## 6. GaussianNB：快速概率基线

高斯朴素贝叶斯假设给定类别后特征条件独立，数值特征满足高斯分布模型。训练很快，可用来讨论假设与数据是否匹配，但不应认为所有 HR 特征都满足假设。

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB(var_smoothing=1e-9)
nb.fit(X_train_ready, y_train)
nb_pred = nb.predict(X_test_ready)
```

`var_smoothing` 用于方差平滑，增大可能改善数值稳定性。BernoulliNB 面向二值输入，CategoricalNB 要求非负整数类别编码，不能直接替换本例包含连续标准化字段的模型。**自己动手**：比较快速基线与其他模型，讨论特征相关性和分布假设。

## 7. 如何接入报告

选择一种模型，将模型对象加入主指南的 `models` 字典。按同样流程完成训练集内部验证、完整训练集重新拟合、测试评价和 ROC/AUC。ROC 分数来源要匹配模型：SVC/LinearSVC 可用 `decision_function`，有 `predict_proba` 的模型检查正类列后使用概率。

报告须说明为什么选它、固定了什么、调了什么、验证结果如何、最终测试表现如何，并解释与两种示例模型的差异。

<!-- website-only:start -->
回到[实验要求]({{ site.baseurl }}/lesson3/2026-11-11-01-lesson3-requirements/)检查交付清单；下一篇为[Lesson 4 房屋价格预测]({{ site.baseurl }}/lesson4/2024-03-05-lesson4-house-price-case/)。
<!-- website-only:end -->

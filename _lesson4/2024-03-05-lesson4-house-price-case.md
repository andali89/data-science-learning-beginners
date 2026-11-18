---
title: 1. 房屋价格预测实验要求
author: Anda Li
date: 2024-03-05
category: Data Science Learning
layout: post
course_previous:
  url: /lesson3/2026-11-11-03-lesson3-models/
  title: 3. 备选分类模型使用方法
course_next:
  url: /lesson4/2024-03-06-lesson4-step-by-step/
  title: 2. 房屋价格预测实施指南
---

本实验使用 1,000 条房屋记录预测房屋售价。你将检查和探索数据、建立回归模型，并结合实际运行结果说明模型在房屋估价中的用途和局限。

<!-- website-only:start -->
## 实验资料

> - <a href="{{ site.baseurl }}/assets/code/lesson4/天津商业大学学生实验报告模板2--管理学院.docx" download>实验报告模板（2026 年，Word）</a>
> - <a href="{{ site.baseurl }}/assets/code/lesson4/《商业数据挖掘》实验指导书2.docx" download>实验指导书（Word）</a>
> - <a href="{{ site.baseurl }}/assets/code/lesson4/codes.zip" download>代码与数据（ZIP）</a>
>
> 解压后打开 `codes` 文件夹，保持 Notebook 与 CSV 在同一目录。
{: .block-tip }

按[实施指南]({{ site.baseurl }}/lesson4/2024-03-06-lesson4-step-by-step/)完成实验。需要更多示例时，可查阅[可视化与回归模型拓展]({{ site.baseurl }}/lesson4/2026-11-18-lesson4-extensions/)。
<!-- website-only:end -->

## 一、实验目的

理解特征和预测目标的含义，以及训练集和测试集各自的用途。比较不同回归方法在同一批房屋上的预测误差，并解释主要特征与房价的关联。

## 二、实验环境

在 Jupyter Notebook 或 PyCharm 中运行实验，使用 pandas、NumPy、Matplotlib、Seaborn 和 scikit-learn。记录实际使用的 Python 和各库版本。实验室使用 Seaborn 0.10.0 时的环境说明见实施指南。

## 三、数据与模型

| 字段 | 含义 |
| --- | --- |
| `Square_Footage` | 居住面积，单位为平方英尺 |
| `Num_Bedrooms` | 卧室数量 |
| `Num_Bathrooms` | 浴室数量 |
| `Year_Built` | 建造年份 |
| `Lot_Size` | 地块面积 |
| `Garage_Size` | 车位数量 |
| `Neighborhood_Quality` | 邻里质量评分，取值 1–10 |
| `House_Price` | 房屋售价，预测目标 |

以多元线性回归为基准，从 CART、SVR 和 MLP 中至少选择一种进行比较。记录实际使用的模型参数。数据文件未明确标注售价的货币单位或地块面积的计量单位，报告可沿用原始数据的单位。

## 四、实验步骤

1. **读取并检查数据。** 核对样本数、字段类型、缺失值和重复记录。
2. **探索性数据分析。** 查看房价分布，用散点图、分组箱线图或相关性热力图观察特征与房价的关系。
3. **准备特征与目标。** 以 2026 年为参考计算房龄，保留房龄并删除建造年份。分离特征矩阵 `X` 和目标 `y`。
4. **划分数据与标准化。** 按 8∶2 将数据划分为训练集和测试集。用训练集的特征拟合 `StandardScaler`，再按同一规则转换两组特征。
5. **训练和比较模型。** 从 MAE、MSE、RMSE 和 R² 中至少选一项，用同一测试集计算各模型的指标并比较结果。需要调参时，在训练集内验证。
6. **解释特征与房价的关联。** 按绝对值对标准化特征的线性回归系数排序。绘图时保留系数的正负号，用来说明关联方向。

## 五、实验结果

使用学校 2026 年的实验报告模板，记录实际运行的关键代码和参数，并呈现探索性分析图表、模型评价指标对比和保留正负号的回归系数图。说明每张图表回答的问题及观察到的结果。可用真实值与预测值对照图展示预测偏差。

## 六、实验意义及讨论

结合自己的实验结果总结主要发现，说明这些发现对房屋估价有何启示，并讨论数据的局限。回归系数反映模型中其他特征保持不变时，该特征与预测房价的关联。若要判断装修或扩建能否提高房价，还需要额外的证据。

> **实验提交**
>
> 按课堂安排，在 **2026 年 11 月 27 日中午 12 点前**将实验报告和代码打包交给学委。压缩包命名为 `学号+姓名+实验报告2`。
{: .block-tip }

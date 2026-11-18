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

本实验用 1,000 条房屋记录预测售价。你将完成数据检查、探索性分析、回归建模与结果解释，并以实际运行结果说明模型在房屋估价中的用途和局限。

<!-- website-only:start -->
## 实验材料

> - <a href="{{ site.baseurl }}/assets/code/lesson4/天津商业大学学生实验报告模板2--管理学院.docx" download>实验报告模板（2026 年，Word）</a>
> - <a href="{{ site.baseurl }}/assets/code/lesson4/《商业数据挖掘》实验指导书2.docx" download>实验指导书（Word）</a>
> - <a href="{{ site.baseurl }}/assets/code/lesson4/codes.zip" download>代码与数据（ZIP）</a>
>
> 解压后打开 `codes` 文件夹，保持 Notebook 与 CSV 在同一目录。
{: .block-tip }

按[实施指南]({{ site.baseurl }}/lesson4/2024-03-06-lesson4-step-by-step/)完成实验；需要更多例子时阅读[可视化与回归模型拓展]({{ site.baseurl }}/lesson4/2026-11-18-lesson4-extensions/)。
<!-- website-only:end -->

## 一、实验目的

理解特征、目标、训练集与测试集的分工，比较不同回归方法在同一批房屋上的预测误差，并解释主要特征与房价的关联。

## 二、实验环境

使用 Jupyter Notebook 或 PyCharm，依赖 pandas、NumPy、Matplotlib、Seaborn 和 scikit-learn。记录实际使用的 Python 与库版本。实验室的 Seaborn 0.10.0 环境说明见实施指南。

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

以多元线性回归为基准，从 CART、SVR、MLP 中至少选择一种进行比较。记录实际使用的参数。数据文件未明确标注售价货币或地块面积的单位，报告可使用原数据单位。

## 四、实验步骤

1. **读取并检查数据。** 核对样本数、字段类型、缺失值和重复记录。
2. **探索性数据分析。** 查看房价分布，用散点图、分组箱线图或相关性热力图观察特征与房价的关系。
3. **准备特征与目标。** 以 2026 年为参考计算房龄，保留房龄并删除建造年份。分离特征矩阵 `X` 和目标 `y`。
4. **划分数据与标准化。** 按 8∶2 划分，随机种子使用 123。用训练特征拟合 `StandardScaler`，再用同一规则转换训练集和测试集。
5. **训练和比较模型。** 在同一测试集上计算至少一种共同指标，选项为 MAE、MSE、RMSE 和 R²。需要调整参数时，在训练集内部验证。
6. **解释特征的关联。** 按标准化特征的线性回归系数绝对值排序，并用保留正负号的图表解释方向。

## 五、实验结果

在 2026 年学校报告模板中放入实际运行的关键代码和参数、探索图表、模型指标对比以及带符号的系数图。用文字解释图表回答的问题和观察到的结果。可用真实值与预测值对照图展示预测偏差。

## 六、实验意义及讨论

根据自己的结果总结主要发现，讨论对房屋估价的启示和数据局限。系数表示控制模型中其他特征后的预测关联。若要判断装修或扩建能否提高房价，还需要额外的证据。

> **实验提交**
>
> 按课堂安排，在 **2026 年 11 月 27 日中午 12 点前**将实验报告和代码打包交给学委。压缩包命名为 `学号+姓名+实验报告2`。
{: .block-tip }

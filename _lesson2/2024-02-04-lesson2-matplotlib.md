---
title: 4. Matplotlib 快速上手：让数据说话
author: Anda Li
date: 2024-02-04
category: Data Science Learning
layout: post
---

“一图胜千言”。在数据分析中，可视化是探索数据、发现规律、呈现结果的最直观、最有力的方式。本章，我们将学习 Python 数据可视化的基石——Matplotlib。

本章，你将学到：
- **Matplotlib 核心概念**：理解 `Figure` (画布) 和 `Axes` (坐标系) 的关系，以及两种绘图 API 的区别。
- **绘制常用图表**: 掌握折线图、柱状图（含水平）、直方图、散点图、饼图和箱形图的绘制。
- **定制图表元素**: 学会添加标题、标签、图例、网格，以及在图表上添加文本和旋转刻度。
- **多子图布局**: 使用 `subplots` 在一张画布上绘制多个图表。
- **保存图表**: 将你的可视化结果保存为图片文件。

> :gift: 温馨提示
>
> 配套资源：本章所有代码和配套资源可以点击如下连接下载 —— <a href="{{ site.baseurl }}/assets/notebooks/lesson2/04-matplotlib.ipynb" target="_blank"  download="04-matplotlib.ipynb">练习 Notebook</a>。下载好后，同学可以在自己的 Anaconda 环境内运行这些代码，这有助于你们快速掌握相关内容。
{: .block-warning }

---

## Matplotlib: 可视化的瑞士军刀

Matplotlib 是 Python 生态中最著名、最基础的可视化库。虽然现在有许多更高级、更美观的库（如 Seaborn, Plotly），但它们中的许多都是基于 Matplotlib 构建的。因此，掌握 Matplotlib 的基本原理，能让你更好地理解和控制几乎所有的 Python 可视化工具。

Matplotlib 的设计理念是让你对图表的每一个元素都有完全的控制权，从坐标轴、刻度、标签到颜色、样式，无所不包。这使得它既能快速绘制简单的图表，也能实现高度定制化的复杂可视化。

> **有用链接**:
> - [Matplotlib 官方网站](https://matplotlib.org/)
> - [Matplotlib 官方图库 (寻找灵感的好地方)](https://matplotlib.org/stable/gallery/index.html)

## 准备工作

通常，我们导入 `matplotlib.pyplot` 模块，并按惯例将其重命名为 `plt`。同时，我们也需要 NumPy 和 Pandas 来准备数据。

在 Jupyter Notebook 中，通常会加上一行“魔法命令” `%matplotlib inline`，它能让图表直接嵌入在 Notebook 的输出单元格中。（在较新版本的 Jupyter 或 VS Code 中，这通常是自动的）。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

plt.rcParams['axes.unicode_minus'] = False

# Jupyter Notebook/IPython 环境下的魔法命令
# %matplotlib inline 

# 设置中文字体（此为示例，具体设置请参考第六章）
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
```

## 一、Matplotlib 的核心：`Figure` 与 `Axes`

在开始画图前，我们需要理解两个核心概念：

1.  **`Figure` (画布)**: 这是最顶层的容器，你可以把它想象成一张空白的画布。一个 `Figure` 对象可以包含一个或多个 `Axes` 对象。
2.  **`Axes` (坐标系/子图)**: 这才是我们真正用来画图的地方，它包含了数据、坐标轴、标签、图例等图表的所有元素。一个 `Axes` 就是一个独立的图表（比如一个折线图或一个柱状图）。

**一个比喻**：`Figure` 就像一个画框，而 `Axes` 则是画框里的一幅幅画。你可以只有一个画框，里面放一幅大画（一个 `Axes`）；也可以有一个大画框，里面用隔板分出好几个区域，每个区域放一幅小画（多个 `Axes`）。

![Matplotlib 的 Figure 与 Axes 结构示意图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_1_figure_axes.png)

最常用的创建 `Figure` 和 `Axes` 的方式是使用 `plt.subplots()`。

> `plt.subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False, constrained_layout=False)`

| 参数 | 说明 |
| --- | --- |
| `nrows`, `ncols` | （可选）子图的行数和列数。 |
| `figsize` | （可选）画布尺寸（宽, 高），单位为英寸。 |
| `sharex`, `sharey` | （可选）是否共享 X/Y 轴，使多个子图刻度一致。 |
| `constrained_layout` | （可选）自动调整子图布局以避免重叠。 |

```python
# 创建一个 Figure 和一个 Axes
# fig 是整个画布对象，ax 是画布中的一个子图（坐标系）对象
fig, ax = plt.subplots(figsize=(6, 4))

# 现在，我们就可以在 ax 上进行绘图了
# (这里我们先不画任何东西，只是展示创建过程)

# 最后，使用 plt.show() 来显示这张画布
plt.show()
```

## 二、一个重要的提醒：两种绘图接口（API）

Matplotlib 有两种使用风格，初学者经常会感到困惑：

1.  **面向对象的接口 (Object-Oriented API)**: 这是我们**强烈推荐并将在本教程中始终使用**的方法。你显式地创建 `Figure` 和 `Axes` 对象，然后调用这些对象的方法来绘图 (e.g., `ax.plot()`, `ax.set_title()`)。它的代码结构清晰，能更好地控制复杂的图表。
2.  **基于状态的接口 (State-based API)**: 这种方式依赖于 `pyplot` 模块 (`plt`) 来“记住”当前正在操作的图表。你直接调用 `plt.plot()`, `plt.title()` 等函数。它的优点是写起来快，适合快速、简单的绘图。

```python
# 准备数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 1. 面向对象风格 (推荐)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("面向对象 (OO) 风格")
plt.show()

# 2. Pyplot 风格 (不推荐用于复杂图表)
plt.figure() # 创建一个全局图表
plt.plot(x, y)
plt.title("Pyplot 风格")
plt.show()
```
当你看到网上的示例直接使用 `plt.plot()` 时，就是在用第二种风格。虽然它也能工作，但我们强烈建议你从一开始就养成使用面向对象接口的习惯，因为它更强大、更不容易出错。

## 三、绘制常见的图表

现在，让我们在坐标系 (`ax`) 上绘制一些最常见的图表。

### 1. 折线图 (`plot`)
**用途**：折线图最适合用来展示数据在一个连续区间（通常是时间）上的**变化趋势**。

> `Axes.plot(x, y, *, color=None, linestyle=None, linewidth=None, marker=None, label=None, alpha=None)`

| 参数 | 说明 |
| --- | --- |
| `x`, `y` | （必选）要绘制的数据序列。 |
| `color` | （可选）线条颜色，可以是颜色字符串或十六进制色值。 |
| `linestyle` | （可选）线型，如 `'-'`, `'--'`, `':'`。 |
| `linewidth` | （可选）线宽，默认 1.5。 |
| `marker` | （可选）数据点标记样式，如 `'o'`, `'s'`。 |
| `label` | （可选）图例标签。 |
| `alpha` | （可选）透明度（0~1）。 |

```python
# 准备数据：x 轴是时间点，y 轴是对应的数值
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)

# 创建 Figure 和 Axes
fig, ax = plt.subplots(figsize=(8, 4))

# 在 ax 上绘制折线图
ax.plot(x_data, y_data, color='steelblue', linewidth=2, marker='o', markevery=10, label='sin(x)')
ax.set_title("简单的正弦函数折线图")
ax.set_xlabel("X 值")
ax.set_ylabel("sin(X)")
ax.legend(loc='best')

# 显示图表
plt.show()
```
![一个简单的正弦函数折线图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_2_line_plot.png)

### 2. 柱状图 (`bar` & `barh`)
**用途**：柱状图用于比较不同**类别**下的数值大小。当类别名称较长时，使用水平柱状图 (`barh`) 会更清晰。

> `Axes.bar(x, height, width=0.8, color=None, label=None, alpha=None)`

| 参数 | 说明 |
| --- | --- |
| `x` | （必选）柱子的类别或位置。 |
| `height` | （必选）柱子的数值高度。 |
| `width` | （可选）柱体宽度。 |
| `color` | （可选）柱子的填充颜色，可以是单个或列表。 |
| `label` | （可选）图例标签。 |
| `alpha` | （可选）透明度。 |

> `Axes.barh(y, width, height=0.8, color=None, label=None, alpha=None)`

| 参数 | 说明 |
| --- | --- |
| `y` | （必选）柱子的类别或位置。 |
| `width` | （必选）水平柱的长度。 |
| `height` | （可选）柱体厚度。 |
| `color`, `label`, `alpha` | （可选）与 `bar` 中含义一致。 |

```python
# 准备数据：类别和对应的数值
categories = ['第一季度', '第二季度', '第三季度', '第四季度']
values = [150, 230, 180, 210]

# 创建一个包含两个子图的 Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# 在左侧子图上绘制垂直柱状图
ax1.bar(categories, values, width=0.6, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], edgecolor='black', label='季度销售')
ax1.set_title('垂直柱状图 (bar)')
ax1.set_ylabel('销售额 (万元)')
ax1.legend()

# 在右侧子图上绘制水平柱状图
ax2.barh(categories, values, height=0.6, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], label='季度销售', alpha=0.8)
ax2.set_title('水平柱状图 (barh)')
ax2.set_xlabel('销售额 (万元)')
ax2.legend(loc='lower right')

fig.suptitle('年度销售额对比')
plt.show()
```
![包含垂直和水平柱状图的对比图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_3_bar_charts.png)

### 3. 直方图 (`hist`)
**用途**：直方图用于展示单个**数值型变量的分布情况**。它会将数据划分成若干个“箱子”（bins），然后统计每个箱子里的数据点数量。

> `Axes.hist(x, bins=None, range=None, density=False, color=None, alpha=None, edgecolor=None)`

| 参数 | 说明 |
| --- | --- |
| `x` | （必选）输入数据（数组或序列）。 |
| `bins` | （可选）分箱数或自定义分箱边界。 |
| `range` | （可选）指定分箱范围 (min, max)。 |
| `density` | （可选）为 `True` 时显示概率密度而非频数。 |
| `color` | （可选）柱子的填充颜色。 |
| `alpha` | （可选）透明度。 |
| `edgecolor` | （可选）柱子边框颜色。 |

```python
# 准备数据：生成 1000 个服从标准正态分布的随机数
data = np.random.randn(1000)

# 创建 Figure 和 Axes
fig, ax = plt.subplots(figsize=(7, 4))

# 在 ax 上绘制直方图，bins=30 表示将数据分成 30 个箱子
ax.hist(data, bins=30, range=(-4, 4), color='#66b3ff', alpha=0.8, edgecolor='black')
ax.set_title("随机数据的分布直方图")
ax.set_xlabel("数值")
ax.set_ylabel("频数")

# 显示图表
plt.show()
```
![一个展示数据分布的直方图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_4_histogram.png)

### 4. 散点图 (`scatter`)
**用途**：散点图用于观察**两个数值型变量之间的关系**（例如，正相关、负相关或不相关）。

> `Axes.scatter(x, y, s=None, c=None, cmap=None, alpha=None, edgecolors=None, label=None)`

| 参数 | 说明 |
| --- | --- |
| `x`, `y` | （必选）点的坐标序列。 |
| `s` | （可选）点的大小（标量或数组）。 |
| `c` | （可选）点的颜色，可为数组表示颜色映射。 |
| `cmap` | （可选）当 `c` 为数值数组时使用的颜色图。 |
| `alpha` | （可选）透明度。 |
| `edgecolors` | （可选）点的边框颜色。 |
| `label` | （可选）图例标签。 |

```python
# 准备数据：两个相关的变量
x_scatter = np.random.rand(50) * 10
# y 与 x 呈线性关系，并加入一些随机噪声
y_scatter = 2 * x_scatter + 1 + np.random.randn(50) * 2

# 创建 Figure 和 Axes
fig, ax = plt.subplots(figsize=(6, 4))

# 在 ax 上绘制散点图
sizes = (np.random.rand(50) * 80) + 20
ax.scatter(x_scatter, y_scatter, s=sizes, c=x_scatter, cmap='viridis', alpha=0.7, edgecolors='black', label='样本点')
ax.set_title("X 和 Y 的散点关系图")
ax.set_xlabel("变量 X")
ax.set_ylabel("变量 Y")
ax.grid(True)
ax.legend(loc='upper left')

# 显示图表
plt.show()
```
![一个展示两个变量线性关系的散点图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_5_scatter.png)

### 5. 饼图 (`pie`)
**用途**：饼图用于展示各个部分占整体的**构成比例**。

**注意**：虽然饼图在商业报告中很常见，但在数据分析领域请谨慎使用。当类别过多或比例相近时，人眼很难准确比较各个扇区的大小。柱状图通常是更好的替代品。

> `Axes.pie(x, explode=None, labels=None, autopct=None, startangle=0, shadow=False, colors=None, pctdistance=0.6)`

| 参数 | 说明 |
| --- | --- |
| `x` | （必选）各部分的数值序列。 |
| `explode` | （可选）用于突出显示的偏移量序列。 |
| `labels` | （可选）每个扇区的标签。 |
| `autopct` | （可选）控制百分比显示的格式字符串。 |
| `startangle` | （可选）起始角度。 |
| `shadow` | （可选）是否绘制阴影。 |
| `colors` | （可选）扇区颜色列表。 |
| `pctdistance` | （可选）百分数字体距离圆心的比例。 |

```python
# 准备数据
labels = ['市场部', '研发部', '销售部', '行政部']
sizes = [15, 30, 45, 10]  # 各部门人数
explode = (0, 0, 0.1, 0)  # '销售部' 稍微突出显示

fig, ax = plt.subplots()

# 绘制饼图
# autopct='%1.1f%%' 会在扇区上显示百分比
colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    shadow=True,
    startangle=90,
    pctdistance=0.8
)

# 保证饼图是正圆的
ax.axis('equal')  
ax.set_title('公司各部门人数占比')

plt.show()
```
![一个显示部门人数占比的饼图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_6_pie.png)

### 6. 箱形图 (`boxplot`)
**用途**：箱形图（或称箱线图）是展示一组或多组数据**分布情况和异常值**的利器。它能清晰地显示出数据的中位数、四分位数、范围和任何潜在的异常点。

> `Axes.boxplot(x, notch=None, vert=True, patch_artist=False, labels=None, showfliers=True, widths=None)`

| 参数 | 说明 |
| --- | --- |
| `x` | （必选）要绘制的数据序列或序列列表。 |
| `notch` | （可选）为 `True` 时绘制凹口箱体，以估计中位数置信区间。 |
| `vert` | （可选）控制箱线垂直 (`True`) 或水平 (`False`)。 |
| `patch_artist` | （可选）允许使用颜色填充箱体。 |
| `tick_labels` | （可选）每个箱子的标签。 |
| `showfliers` | （可选）是否显示异常值（离群点）。 |
| `widths` | （可选）控制每个箱子的宽度。 |

```python
# 准备数据：三组不同分布的数据
np.random.seed(10)
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(80, 30, 200)
data3 = np.random.normal(90, 20, 200)
data_to_plot = [data1, data2, data3]

fig, ax = plt.subplots()

# 绘制箱形图
bp = ax.boxplot(
    data_to_plot,
    patch_artist=True,
    labels=['A产品', 'B产品', 'C产品'],
    showfliers=True,
    widths=0.6
)

# 为箱子填充颜色
colors = ['#99ff99', '#66b3ff', '#ff9999']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_title('不同产品销售额分布对比')
ax.set_ylabel('销售额')
ax.yaxis.grid(True)

plt.show()
```
![一个对比三组数据分布的箱形图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_7_boxplot.png)

## 四、定制你的图表：让信息更清晰

一个“裸”图表很难传递有效信息。我们需要为它添加标题、标签、图例等元素，让它变得清晰易读。

### 1. 添加标题、标签和图例

| 方法 | 常用参数 | 说明 |
| --- | --- | --- |
| `Axes.set_title(label, loc='center', pad=None)` | `label`（必选）; `loc`（可选）: 标题位置 (`'left'`, `'center'`, `'right'`); `pad`（可选）: 标题与图表的距离 | 设置子图标题。 |
| `Axes.set_xlabel(label, fontsize=None)` | `label`（必选）; `fontsize`（可选）: 字体大小 | 设置 X 轴标签。 |
| `Axes.set_ylabel(label, fontsize=None)` | `label`（必选）; `fontsize`（可选）: 字体大小 | 设置 Y 轴标签。 |
| `Axes.legend(loc='best', ncol=1, frameon=True)` | `loc`（可选）: 图例位置；`ncol`（可选）: 列数；`frameon`（可选）: 是否显示边框 | 添加图例。 |

```python
# 准备数据
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建 Figure 和 Axes
fig, ax = plt.subplots()

# 绘制两条折线，并为它们设置 label，用于后续生成图例
ax.plot(x, y1, label='sin(x)')
ax.plot(x, y2, label='cos(x)', linestyle='--')

# 1. 添加标题
ax.set_title('正弦与余弦函数图像', loc='left', pad=12)

# 2. 添加 X 轴和 Y 轴的标签
ax.set_xlabel('X 轴 (弧度)', fontsize=11)
ax.set_ylabel('Y 轴 (值)', fontsize=11)

# 3. 添加图例 (它会自动寻找 plot 中设置的 label)
ax.legend(loc='best', ncol=2)

plt.show()
```
![一个包含标题、坐标轴标签和图例的图表]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_8_labels_legend.png)

### 2. 调整样式、颜色和网格

| 方法 | 常用参数 | 说明 |
| --- | --- | --- |
| `Axes.grid(visible=True, which='major', axis='both', linestyle='--', alpha=None)` | `visible`（可选）控制网格可见性；`which`（可选）网格类型；`axis`（可选）作用轴；`linestyle`（可选）样式；`alpha`（可选）透明度。 |
| `Axes.set_xlim(left=None, right=None)` | `left`（可选）和 `right`（可选）设置 X 轴范围。 |
| `Axes.set_ylim(bottom=None, top=None)` | `bottom`（可选）和 `top`（可选）设置 Y 轴范围。 |
| `Axes.plot(..., color=None, linestyle=None, marker=None, linewidth=None)` | `color`（可选）、`linestyle`（可选）、`marker`（可选）、`linewidth`（可选）自定义线条样式与标记。 |

```python
fig, ax = plt.subplots()

# 绘制时指定颜色、线型、标记样式
ax.plot(x, y1, label='sin(x)', color='blue', linestyle='-', marker='o', markersize=3, linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linestyle=':', linewidth=2)

# 添加网格
ax.grid(visible=True, linestyle=':', alpha=0.6)

# 设置坐标轴范围
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)

ax.set_title('定制化样式的图表')
ax.legend()
plt.show()
```
![一个经过精细定制的、包含两条曲线的折线图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_9_custom_style.png)

### 3. 添加文本注解和旋转刻度
有时我们需要在图表的特定位置高亮数据，或者当坐标轴标签太长而重叠时，需要对它们进行调整。

| 方法 | 常用参数 | 说明 |
| --- | --- | --- |
| `Axes.tick_params(axis='x', rotation=None, labelsize=None)` | `axis`（必选）指定作用轴；`rotation`（可选）刻度方向；`labelsize`（可选）标签大小。 |
| `Axes.text(x, y, s, ha='center', va='center', fontsize=None)` | `x`（必选）、`y`（必选）位置；`s`（必选）文本内容；`ha`（可选）水平对齐；`va`（可选）垂直对齐；`fontsize`（可选）字号。 |
| `Figure.tight_layout(pad=1.08)` | `pad`（可选）调整整个画布的布局，避免元素重叠。 |

```python
# 准备数据
categories = ['非常不满意', '不满意', '一般', '满意', '非常满意']
values = [5, 25, 50, 120, 200]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, values, color='#66b3ff')

# 1. 旋转 X 轴的刻度标签，以防重叠
ax.tick_params(axis='x', rotation=45)

# 2. 在每个柱子顶端添加数值标签
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval,
        f"{int(yval)}",
        va='bottom',
        ha='center',
        fontsize=10,
        fontweight='bold'
    ) 

ax.set_title('客户满意度调查结果')
ax.set_ylabel('投票数')
fig.tight_layout() # 调整布局以防标签被截断
plt.show()
```
![一个带有数值标签和旋转刻度的柱状图]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_10_bar_annotations.png)

## 五、多子图布局 (`subplots`)

当你想将多个相关的图表并排比较时，就需要用到多子图布局。`plt.subplots()` 函数是实现这一点的最佳方式。

| 参数 | 说明 |
| --- | --- |
| `nrows`, `ncols` | （可选）子图网格的行列数。 |
| `figsize` | （可选）整个画布大小。 |
| `sharex`, `sharey` | （可选）在多个子图之间共享坐标轴。 |
| `squeeze` | （可选）控制返回的 `axes` 维度，`False` 时总是返回二维数组。 |
| `gridspec_kw` | （可选）传递给底层 `GridSpec` 的字典，可细致控制布局。 |

```python
# 创建一个 2x2 的子图网格
# fig 是整个画布，axes 是一个 2x2 的 NumPy 数组，每个元素都是一个子图对象
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), constrained_layout=True)

# --- 在每个子图上绘图 ---

# 左上角子图 (axes[0, 0])
axes[0, 0].plot(np.sin(np.linspace(0, 10, 100)))
axes[0, 0].set_title('折线图')

# 右上角子图 (axes[0, 1])
axes[0, 1].bar(['A', 'B', 'C'], [3, 5, 2])
axes[0, 1].set_title('柱状图')

# 左下角子图 (axes[1, 0])
axes[1, 0].hist(np.random.randn(500), bins=20)
axes[1, 0].set_title('直方图')

# 右下角子图 (axes[1, 1])
axes[1, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[1, 1].set_title('散点图')

# 显示整个画布
plt.show()
```
![一个 2x2 的子图网格，包含四种不同类型的图表]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_11_subplots.png)

## 六、保存图表

将图表保存到文件中非常简单，只需使用 `fig.savefig()` 方法。

| 方法 | 常用参数 | 说明 |
| --- | --- | --- |
| `Figure.savefig(fname, dpi=None, bbox_inches=None, facecolor=None, transparent=False)` | `fname`（必选）: 输出路径；`dpi`（可选）: 分辨率；`bbox_inches`（可选）: 调整图像边界；`facecolor`（可选）: 背景颜色；`transparent`（可选）: 是否透明背景。 |

```python
from pathlib import Path

output_path = Path("assets/img_ana/lesson2/matplotlib")
output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
ax.plot(x, y1, label='sin(x)')
ax.set_title('保存这张图')
ax.legend()

save_path = output_path / "figure_4_12_saved_plot.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)

print(f"图表已保存为 {save_path}")
```
![保存图表示例]({{ site.baseurl }}/assets/img_ana/lesson2/matplotlib/figure_4_12_saved_plot.png)

> **关于中文显示问题**
> Matplotlib 默认的字体不支持中文，因此如果你在图表中直接使用中文，可能会显示为方框 `□`。这是一个常见问题，我们将在 **第六章：常见问题与排查** 中提供详细的解决方案。

---
**实践小结**: 你已经学会了使用 Matplotlib 绘制和定制最常见的几种基本图表，并且掌握了如何将它们组合在同一张画布上以及如何将结果保存。可视化是数据故事叙述的开始。下一章，我们将把前面三章所学的 NumPy, Pandas, Matplotlib 知识串联起来，完成一个从数据加载、处理到可视化的端到端小案例。

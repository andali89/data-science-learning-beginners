---
title: 4.1 Matplotlib 常用操作速查表
author: Anda Li
date: 2024-02-04 12:00:00 +0800
category: Data Science Learning
layout: post
---

<div class="cheat-sheet">
  <div class="cheat-header">
    <h1>Matplotlib 常用操作速查表</h1>
    <div class="cheat-subtitle">面向初学者 · Figure / Axes、常用图表、标注与布局 · 支持 A4 打印</div>
  </div>

  <div class="cheat-toolbar">
    <button class="cheat-print-button" onclick="window.print()">打印 / 保存为 PDF</button>
  </div>

  <div class="cheat-callout">
    常用导入方式：<code>import matplotlib.pyplot as plt</code>。课堂示例优先使用面向对象写法：<code>fig, ax = plt.subplots()</code>，再通过 <code>ax</code> 完成绘图和设置。
  </div>

  <div class="cheat-grid">
    <section class="cheat-card">
      <h2>创建 Figure 与 Axes</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>fig, ax = plt.subplots()</code></td><td>创建一个 Figure 和一个 Axes</td></tr>
          <tr><td><code>plt.subplots(figsize=(8, 5))</code></td><td>指定图形尺寸</td></tr>
          <tr><td><code>fig, axs = plt.subplots(2, 2)</code></td><td>创建 2×2 子图</td></tr>
          <tr><td><code>plt.show()</code></td><td>显示图形</td></tr>
          <tr><td><code>ax.clear()</code></td><td>清空当前 Axes</td></tr>
          <tr><td><code>fig.clear()</code></td><td>清空整个 Figure</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>常用图表</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>ax.plot(x, y)</code></td><td>折线图</td></tr>
          <tr><td><code>ax.scatter(x, y)</code></td><td>散点图</td></tr>
          <tr><td><code>ax.bar(x, y)</code></td><td>垂直柱状图</td></tr>
          <tr><td><code>ax.barh(y, x)</code></td><td>水平柱状图</td></tr>
          <tr><td><code>ax.hist(x, bins=10)</code></td><td>直方图</td></tr>
          <tr><td><code>ax.boxplot(x)</code></td><td>箱形图</td></tr>
          <tr><td><code>ax.pie(values, labels=labels)</code></td><td>饼图</td></tr>
          <tr><td><code>ax.imshow(image)</code></td><td>显示图像或二维矩阵</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>标题、坐标轴与图例</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>ax.set_title('Title')</code></td><td>设置图标题</td></tr>
          <tr><td><code>ax.set_xlabel('X')</code></td><td>设置 x 轴标签</td></tr>
          <tr><td><code>ax.set_ylabel('Y')</code></td><td>设置 y 轴标签</td></tr>
          <tr><td><code>ax.set_xlim(0, 10)</code></td><td>设置 x 轴显示范围</td></tr>
          <tr><td><code>ax.set_ylim(0, 100)</code></td><td>设置 y 轴显示范围</td></tr>
          <tr><td><code>ax.legend()</code></td><td>显示图例</td></tr>
          <tr><td><code>ax.grid(True)</code></td><td>显示网格线</td></tr>
          <tr><td><code>ax.tick_params(axis='x', rotation=45)</code></td><td>旋转 x 轴刻度标签</td></tr>
          <tr><td><code>ax.set_xticks(ticks)</code></td><td>设置 x 轴刻度位置</td></tr>
          <tr><td><code>ax.set_xticklabels(labels)</code></td><td>设置 x 轴刻度文字</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>标注、布局与保存</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>ax.text(x, y, 'text')</code></td><td>在指定坐标添加文字</td></tr>
          <tr><td><code>ax.annotate('note', xy=(x, y))</code></td><td>为数据点添加注释</td></tr>
          <tr><td><code>fig.suptitle('Title')</code></td><td>设置整张 Figure 的总标题</td></tr>
          <tr><td><code>fig.tight_layout()</code></td><td>自动调整子图间距</td></tr>
          <tr><td><code>plt.subplots(constrained_layout=True)</code></td><td>创建时启用自动布局</td></tr>
          <tr><td><code>axs[0, 1].plot(x, y)</code></td><td>在指定子图上绘图</td></tr>
          <tr><td><code>fig.savefig('figure.png', dpi=300)</code></td><td>保存高分辨率图片</td></tr>
          <tr><td><code>fig.savefig('figure.pdf')</code></td><td>保存为矢量 PDF</td></tr>
          <tr><td><code>plt.close(fig)</code></td><td>关闭 Figure，批量绘图时可释放资源</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <div class="cheat-note">
    课堂提示：先区分 <code>Figure</code>（整张画布）和 <code>Axes</code>（具体坐标区域）。多数绘图和标题、坐标轴设置都在 <code>ax</code> 上完成；保存整张图时使用 <code>fig.savefig()</code>。
  </div>
</div>

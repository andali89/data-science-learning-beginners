---
title: 2.1 NumPy 常用操作速查表
author: Anda Li
date: 2024-02-02 12:00:00 +0800
category: Data Science Learning
layout: post
---

<div class="cheat-sheet">
  <div class="cheat-header">
    <h1>NumPy 常用操作速查表</h1>
    <div class="cheat-subtitle">面向初学者 · 数组、索引、变形、统计与常用计算 · 支持 A4 打印</div>
  </div>

  <div class="cheat-toolbar">
    <button class="cheat-print-button" onclick="window.print()">打印 / 保存为 PDF</button>
  </div>

  <div class="cheat-callout">
    常用导入方式：<code>import numpy as np</code>。NumPy 的核心对象是 <code>ndarray</code>，多数操作都围绕数组的形状、索引和向量化计算展开。
  </div>

  <div class="cheat-grid">
    <section class="cheat-card">
      <h2>创建与查看数组</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>np.array([1, 2, 3])</code></td><td>由列表创建数组</td></tr>
          <tr><td><code>np.arange(0, 10, 2)</code></td><td>按步长生成序列</td></tr>
          <tr><td><code>np.linspace(0, 1, 5)</code></td><td>在区间内等距生成指定数量的值</td></tr>
          <tr><td><code>np.zeros((2, 3))</code></td><td>创建全 0 数组</td></tr>
          <tr><td><code>np.ones((2, 3))</code></td><td>创建全 1 数组</td></tr>
          <tr><td><code>np.full((2, 3), 7)</code></td><td>创建指定常数填充的数组</td></tr>
          <tr><td><code>np.eye(3)</code></td><td>创建单位矩阵</td></tr>
          <tr><td><code>a.shape</code></td><td>查看数组形状</td></tr>
          <tr><td><code>a.ndim</code></td><td>查看维数</td></tr>
          <tr><td><code>a.dtype</code></td><td>查看元素数据类型</td></tr>
          <tr><td><code>a.size</code></td><td>查看元素总数</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>索引、筛选与形状变换</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>a[0]</code></td><td>取第一个元素或第一行</td></tr>
          <tr><td><code>a[:, 1]</code></td><td>取二维数组第 2 列</td></tr>
          <tr><td><code>a[1:4]</code></td><td>切片，取索引 1 到 3</td></tr>
          <tr><td><code>a[a &gt; 0]</code></td><td>布尔索引，筛选满足条件的元素</td></tr>
          <tr><td><code>a.reshape(2, 3)</code></td><td>改变形状，不改变元素总数</td></tr>
          <tr><td><code>a.ravel()</code></td><td>将数组展平为一维，尽可能返回视图</td></tr>
          <tr><td><code>a.flatten()</code></td><td>展平为一维，并返回副本</td></tr>
          <tr><td><code>a.T</code></td><td>转置数组</td></tr>
          <tr><td><code>a[:, np.newaxis]</code></td><td>增加一个维度</td></tr>
          <tr><td><code>np.concatenate([a, b])</code></td><td>沿已有轴拼接数组</td></tr>
          <tr><td><code>np.stack([a, b])</code></td><td>沿新轴堆叠数组</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>计算与统计</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>a + b</code></td><td>逐元素加法，可结合广播机制</td></tr>
          <tr><td><code>a * b</code></td><td>逐元素乘法</td></tr>
          <tr><td><code>a @ b</code></td><td>矩阵乘法</td></tr>
          <tr><td><code>np.sum(a)</code></td><td>求和</td></tr>
          <tr><td><code>np.mean(a)</code></td><td>求均值</td></tr>
          <tr><td><code>np.std(a)</code></td><td>求标准差</td></tr>
          <tr><td><code>np.min(a)</code> / <code>np.max(a)</code></td><td>求最小值 / 最大值</td></tr>
          <tr><td><code>np.sum(a, axis=0)</code></td><td>沿指定轴计算，例如按列求和</td></tr>
          <tr><td><code>np.where(a &gt; 0, 1, 0)</code></td><td>按条件选择或赋值</td></tr>
          <tr><td><code>np.clip(a, 0, 100)</code></td><td>将值限制在指定区间</td></tr>
          <tr><td><code>np.unique(a)</code></td><td>返回不重复值</td></tr>
          <tr><td><code>np.sort(a)</code></td><td>返回排序后的数组</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>随机数、缺失值与保存</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>rng = np.random.default_rng(42)</code></td><td>创建推荐的随机数生成器，并设置随机种子</td></tr>
          <tr><td><code>rng.random(5)</code></td><td>生成 0 到 1 之间的随机数</td></tr>
          <tr><td><code>rng.integers(0, 10, 5)</code></td><td>生成随机整数</td></tr>
          <tr><td><code>np.isnan(a)</code></td><td>判断元素是否为 <code>NaN</code></td></tr>
          <tr><td><code>np.nanmean(a)</code></td><td>忽略 <code>NaN</code> 计算均值</td></tr>
          <tr><td><code>b = a.copy()</code></td><td>显式复制数组，避免共享底层数据</td></tr>
          <tr><td><code>np.allclose(a, b)</code></td><td>判断两个数组在数值误差范围内是否接近</td></tr>
          <tr><td><code>np.linalg.norm(a)</code></td><td>计算向量或矩阵范数</td></tr>
          <tr><td><code>np.linalg.solve(A, b)</code></td><td>求解线性方程组 <code>Ax=b</code></td></tr>
          <tr><td><code>np.save('a.npy', a)</code></td><td>保存单个 NumPy 数组</td></tr>
          <tr><td><code>np.load('a.npy')</code></td><td>读取 NumPy 数组文件</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <div class="cheat-note">
    课堂提示：重点掌握 <code>shape</code>、索引与切片、<code>axis</code>、<code>reshape</code>、布尔索引和广播。遇到结果维度不符合预期时，先检查 <code>a.shape</code>。
  </div>
</div>

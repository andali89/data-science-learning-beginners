---
title: Pandas 常用操作速查表
nav_title: 附录：Pandas 速查表
author: Anda Li
date: 2024-02-03 12:00:00 +0800
category: Data Science Learning
layout: post
hide_title: true
hide_sidebar_toc: true
---

<div class="cheat-sheet">
  <div class="cheat-header">
    <h1>Pandas 常用操作速查表</h1>
    <div class="cheat-subtitle">面向初学者 · DataFrame、筛选、清洗、分组与合并 · 支持 A4 打印</div>
  </div>

  <div class="cheat-toolbar">
    <button class="cheat-print-button" onclick="window.print()">打印 / 保存为 PDF</button>
  </div>

  <div class="cheat-callout">
    常用导入方式：<code>import pandas as pd</code>。分析表格数据时，建议先用 <code>head()</code>、<code>shape</code>、<code>info()</code> 和 <code>describe()</code> 熟悉数据。
  </div>

  <div class="cheat-grid">
    <section class="cheat-card">
      <h2>读取、创建与查看数据</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>pd.read_csv('data.csv')</code></td><td>读取 CSV 文件</td></tr>
          <tr><td><code>pd.read_excel('data.xlsx')</code></td><td>读取 Excel 文件</td></tr>
          <tr><td><code>pd.DataFrame(data)</code></td><td>创建 DataFrame</td></tr>
          <tr><td><code>df.head()</code></td><td>查看前 5 行</td></tr>
          <tr><td><code>df.tail()</code></td><td>查看后 5 行</td></tr>
          <tr><td><code>df.shape</code></td><td>查看行数和列数</td></tr>
          <tr><td><code>df.columns</code></td><td>查看列名</td></tr>
          <tr><td><code>df.dtypes</code></td><td>查看各列数据类型</td></tr>
          <tr><td><code>df.info()</code></td><td>查看结构、非空值数量和数据类型</td></tr>
          <tr><td><code>df.describe()</code></td><td>查看数值列的描述性统计</td></tr>
          <tr><td><code>df['col'].value_counts()</code></td><td>统计不同取值及其频数</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>选择、筛选与排序</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>df['col']</code></td><td>选择单列，返回 Series</td></tr>
          <tr><td><code>df[['a', 'b']]</code></td><td>选择多列，返回 DataFrame</td></tr>
          <tr><td><code>df.loc[0:3, ['a', 'b']]</code></td><td>按标签选择行和列</td></tr>
          <tr><td><code>df.iloc[0:3, 0:2]</code></td><td>按整数位置选择行和列</td></tr>
          <tr><td><code>df[df['age'] &gt; 20]</code></td><td>按条件筛选行</td></tr>
          <tr><td><code>df[(c1) &amp; (c2)]</code></td><td>同时满足两个条件</td></tr>
          <tr><td><code>df[(c1) | (c2)]</code></td><td>满足任一条件</td></tr>
          <tr><td><code>df['col'].isin(values)</code></td><td>判断值是否属于给定集合</td></tr>
          <tr><td><code>df.sort_values('col')</code></td><td>按某列排序</td></tr>
          <tr><td><code>df.sort_values('col', ascending=False)</code></td><td>按某列降序排序</td></tr>
          <tr><td><code>df.nlargest(5, 'score')</code></td><td>取某列最大的前 5 行</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>数据清洗与转换</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>df.isna().sum()</code></td><td>统计各列缺失值数量</td></tr>
          <tr><td><code>df.dropna()</code></td><td>删除含缺失值的行</td></tr>
          <tr><td><code>df['col'].fillna(value)</code></td><td>填充某列缺失值</td></tr>
          <tr><td><code>df.duplicated().sum()</code></td><td>统计重复行数量</td></tr>
          <tr><td><code>df.drop_duplicates()</code></td><td>删除重复行</td></tr>
          <tr><td><code>df.rename(columns={'a':'A'})</code></td><td>重命名列</td></tr>
          <tr><td><code>df['col'].astype('int64')</code></td><td>转换数据类型</td></tr>
          <tr><td><code>pd.to_datetime(df['date'])</code></td><td>转换为日期时间类型</td></tr>
          <tr><td><code>df['new'] = df['a'] + df['b']</code></td><td>根据已有列创建新列</td></tr>
          <tr><td><code>df['col'].map(mapping)</code></td><td>按映射关系替换值</td></tr>
          <tr><td><code>df['col'].str.strip()</code></td><td>去除字符串首尾空格</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>分组、合并、重塑与导出</h2>
      <table>
        <thead><tr><th>写法</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><code>df.groupby('group')['x'].mean()</code></td><td>按组计算均值</td></tr>
          <tr><td><code>df.groupby('group').agg(...)</code></td><td>对各组执行多个聚合统计</td></tr>
          <tr><td><code>pd.concat([df1, df2])</code></td><td>按行或按列拼接数据</td></tr>
          <tr><td><code>pd.merge(df1, df2, on='id')</code></td><td>按共同键连接两个表</td></tr>
          <tr><td><code>pd.merge(..., how='left')</code></td><td>左连接，保留左表全部记录</td></tr>
          <tr><td><code>df.pivot_table(...)</code></td><td>创建透视表并进行聚合</td></tr>
          <tr><td><code>df.melt(...)</code></td><td>将宽表转换为长表</td></tr>
          <tr><td><code>df.reset_index()</code></td><td>将索引恢复为普通列</td></tr>
          <tr><td><code>df.set_index('id')</code></td><td>将指定列设置为索引</td></tr>
          <tr><td><code>df.to_csv('out.csv', index=False)</code></td><td>保存为 CSV</td></tr>
          <tr><td><code>df.to_excel('out.xlsx', index=False)</code></td><td>保存为 Excel</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <div class="cheat-note">
    课堂提示：先分清 <code>loc</code>（按标签）与 <code>iloc</code>（按位置）；筛选多个条件时，每个条件通常需要加括号，并用 <code>&amp;</code> 或 <code>|</code> 连接。
  </div>
</div>

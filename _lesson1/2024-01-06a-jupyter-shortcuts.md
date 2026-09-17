---
title: Jupyter Notebook 常用快捷键速查表
nav_title: 附录：Jupyter 快捷键速查表
author: Anda Li
date: 2024-01-06 12:00:00 +0800
category: Data Science Learning
layout: post
hide_title: true
hide_sidebar_toc: true
---

<div class="cheat-sheet">
  <div class="cheat-header">
    <h1>Jupyter Notebook / JupyterLab 常用快捷键</h1>
    <div class="cheat-subtitle">中文速查表 · Windows / Linux · 适合课堂使用与 A4 打印</div>
  </div>

  <div class="cheat-toolbar">
    <button class="cheat-print-button" onclick="window.print()">打印 / 保存为 PDF</button>
  </div>

  <div class="cheat-callout">
    <strong>两种模式：</strong>
    <kbd>Esc</kbd> → 命令模式（操作单元格）　|　<kbd>Enter</kbd> → 编辑模式（编辑单元格内容）
  </div>

  <div class="cheat-grid">
    <section class="cheat-card">
      <h2>运行与保存</h2>
      <table>
        <thead><tr><th>快捷键</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><kbd>Shift</kbd> + <kbd>Enter</kbd></td><td><strong>运行当前单元格，并移动到下一个单元格</strong></td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>Enter</kbd></td><td>运行当前单元格，并停留在当前单元格</td></tr>
          <tr><td><kbd>Alt</kbd> + <kbd>Enter</kbd></td><td>运行当前单元格，并在下方新建单元格</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>S</kbd></td><td>保存 Notebook</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>编辑模式</h2>
      <table>
        <thead><tr><th>快捷键</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><kbd>Ctrl</kbd> + <kbd>A</kbd></td><td>选中当前单元格中的全部文本</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>Z</kbd></td><td>撤销文本编辑</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>Y</kbd></td><td>重做文本编辑（部分环境可能不同）</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>/</kbd></td><td>注释 / 取消注释所选代码</td></tr>
          <tr><td><kbd>Tab</kbd></td><td>缩进或触发代码补全</td></tr>
          <tr><td><kbd>Shift</kbd> + <kbd>Tab</kbd></td><td>查看函数签名或帮助信息</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>]</kbd></td><td>增加缩进</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>[</kbd></td><td>减少缩进</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>命令模式：单元格操作</h2>
      <table>
        <thead><tr><th>快捷键</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td><kbd>A</kbd></td><td>在当前单元格上方新增单元格</td></tr>
          <tr><td><kbd>B</kbd></td><td>在当前单元格下方新增单元格</td></tr>
          <tr><td><kbd>D</kbd> <kbd>D</kbd></td><td>删除当前单元格</td></tr>
          <tr><td><kbd>Z</kbd></td><td>撤销删除单元格</td></tr>
          <tr><td><kbd>M</kbd></td><td>将单元格转换为 Markdown</td></tr>
          <tr><td><kbd>Y</kbd></td><td>将单元格转换为 Code</td></tr>
          <tr><td><kbd>C</kbd></td><td>复制单元格</td></tr>
          <tr><td><kbd>X</kbd></td><td>剪切单元格</td></tr>
          <tr><td><kbd>V</kbd></td><td>在下方粘贴单元格</td></tr>
          <tr><td><kbd>Shift</kbd> + <kbd>V</kbd></td><td>在上方粘贴单元格</td></tr>
          <tr><td><kbd>Shift</kbd> + <kbd>M</kbd></td><td>合并选中的单元格</td></tr>
          <tr><td><kbd>H</kbd></td><td>查看完整快捷键列表</td></tr>
        </tbody>
      </table>
    </section>

    <section class="cheat-card">
      <h2>课堂最常用的快捷键</h2>
      <table>
        <thead><tr><th>快捷键</th><th>记住它的用途</th></tr></thead>
        <tbody>
          <tr><td><kbd>Shift</kbd> + <kbd>Enter</kbd></td><td>运行当前单元格</td></tr>
          <tr><td><kbd>Esc</kbd></td><td>进入命令模式</td></tr>
          <tr><td><kbd>Enter</kbd></td><td>进入编辑模式</td></tr>
          <tr><td><kbd>A</kbd> / <kbd>B</kbd></td><td>在上方 / 下方新增单元格</td></tr>
          <tr><td><kbd>D</kbd> <kbd>D</kbd></td><td>删除单元格</td></tr>
          <tr><td><kbd>M</kbd> / <kbd>Y</kbd></td><td>切换 Markdown / Code</td></tr>
          <tr><td><kbd>Ctrl</kbd> + <kbd>S</kbd></td><td>保存 Notebook</td></tr>
          <tr><td><kbd>H</kbd></td><td>忘记快捷键时查看完整列表</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <div class="cheat-note">
    提示：命令模式下的字母快捷键需要先按 <kbd>Esc</kbd>。不同 Jupyter Notebook / JupyterLab 版本或浏览器设置可能对少数快捷键有所调整，可按 <kbd>H</kbd> 查看当前环境中的完整列表。
  </div>
</div>

---
title: 6. Jupyter Notebook 快速入门：你的第一个数据分析笔记
author: Anda Li
date: 2024-01-06
category: Data Science Learning
layout: post
---


理论说了这么多，现在是时候亲自动手，体验一下 Jupyter Notebook 的魅力了。我们将通过一个非常简单的“玩具案例”来熟悉 Jupyter 的基本操作。

## 启动 Jupyter Notebook

再次回顾一下启动步骤：

1.  **为项目创建文件夹：** 在你的电脑上（比如桌面）创建一个新的文件夹，命名为 `my_first_notebook`。
2.  **打开命令行并进入该文件夹：**
    *   打开命令行终端（`cmd` 或 `Terminal`）。
    *   使用 `cd` 命令进入你刚刚创建的文件夹。例如：

    ```bash
    # Windows (cmd 或 PowerShell)
    cd %USERPROFILE%\Desktop\my_first_notebook
    
    ```

3.  **启动 Jupyter：** 在该目录下运行 `jupyter notebook`（如上代码块所示）。
      ```bash
    # Windows (cmd 或 PowerShell)
    jupyter notebook
    ```


你的浏览器会自动打开，显示 `my_first_notebook` 文件夹の内容（现在应该是空的）。

## 创建你的第一个 Notebook

1.  在 Jupyter 的主界面右上角，点击 **“New”** 按钮。
2.  在下拉菜单中，选择 **“Python 3 (ipykernel)”**（或者类似的选项）。

    ![Jupyter 主界面：点击 "New" 并选择 "Python 3"]({{ site.baseurl}}/assets/img_ana/Section6_jupyter_main.png)

    *图：Jupyter 主界面截图*

3.  浏览器会打开一个新的标签页，这就是你的 Notebook 界面！默认标题是 `Untitled.ipynb`。`.ipynb` 是 “IPython Notebook” 的缩写，是 Jupyter Notebook 文件的标准扩展名。

4.  **重命名文件：** 点击顶部的 “Untitled”，在弹出的对话框中输入一个新的名字，比如 `hello_jupyter`，然后点击 “Rename”。

    ![重命名 Notebook：在标题上点击并输入新名称]({{ site.baseurl}}/assets/img_ana/Section6_jupyter_rename.png)

    *图：Notebook 界面截图*

## 认识 Notebook 界面

你的 Notebook 界面主要由以下几个部分组成：

*   **菜单栏：** 提供文件操作、编辑、查看、插入单元格等功能。
*   **工具栏：** 提供常用操作的快捷按钮，如保存、添加单元格、剪切、复制、运行等。
*   **单元格 (Cell)：** 这是 Notebook 的核心！你的所有代码和文本都将写在单元格里。

## 单元格的两种模式

Jupyter 的单元格有两种模式，理解这一点至关重要：

1.  **命令模式 (Command Mode):**
    *   **标志：** 单元格的边框是**浅色**的。
    *   **进入方式：** 按 `Esc` 键，或者用鼠标点击单元格的左侧空白区域。
    *   **功能：** 在这个模式下，你可以对单元格本身进行操作，比如创建、删除、移动、复制单元格等。你输入的按键会被当作快捷键命令。

2.  **编辑模式 (Edit Mode):**
    *   **标志：** 单元格的边框是**亮色**的，并且内部有光标在闪烁。
    *   **进入方式：** 按 `Enter` 键，或者用鼠标点击单元格的代码区域。
    *   **功能：** 在这个模式下，你可以在单元格内编写代码或文本。

**在两种模式间切换是最高频的操作，请务必记住：**

> `Esc` 进入命令模式，`Enter` 进入编辑模式。
{: .block-tip }

## 开始我们的“玩具案例”

### 1. Code 单元格：运行你的第一行 Python 代码

默认创建的单元格就是代码（Code）单元格。

1.  在第一个单元格中，输入以下 Python 代码：

    ```python
    print("Hello, Jupyter!")
    ```

2.  **运行单元格：** 按下 `Shift + Enter`。这是运行单元格并自动选择（或创建）下一个单元格的快捷键，也是你用得最多的快捷键。

3.  你会立刻在单元格下方看到代码的输出：`Hello, Jupyter!`。

     
    ![单元格示例：代码与输出]({{ site.baseurl}}/assets/img_ana/Section6_jupyter_print.png)

    *单元格示例：代码与输出*

### 2. Markdown 单元格：写下你的笔记

Jupyter 的强大之处在于它可以将代码和说明文字无缝结合。现在，我们来添加一些说明。

1.  你的光标现在应该在一个新的单元格里。如果不是，可以在命令模式下按 `B` 键（Below）来创建一个新的单元格。
2.  **切换为 Markdown 模式：** 确保单元格处于**命令模式**（边框是蓝色的），然后按一下 `M` 键 (使用 `shift+m` 会失败，使用`Caps Lock` 切换后再输入 `M`)。你会发现单元格左侧的 `In [ ]:` 标记消失了。
3.  **进入编辑模式**（按 `Enter`），然后输入以下文本：

    ```markdown
    # 我的第一个数据分析笔记

    这是一个简单的例子，用来计算一个列表里所有数字的和。

    我们将执行以下步骤：
    1.  创建一个包含数字的列表。
    2.  使用 `sum()` 函数计算总和。
    3.  打印结果。
    ```

4.  **渲染 Markdown：** 同样，按下 `Shift + Enter` 来“运行”这个 Markdown 单元格。你会看到格式化的文本，而不是原始的 Markdown 代码。

    ![渲染后的 Markdown 文本截图：漂亮的标题和列表]({{ site.baseurl}}/assets/img_ana/Section6_jupyter_md.png)

    *图：渲染后的 Markdown 文本截图，显示漂亮的标题和列表。*

### 3. 结合代码与 Markdown

现在，让我们按照我们刚刚写的笔记来执行代码。

1.  在新的 Code 单元格中，输入以下代码：

    ```python
    # 1. 创建一个包含数字的列表
    my_list = [10, 25, 30, 45, 50]

    # 2. 使用 sum() 函数计算总和
    total = sum(my_list)

    # 3. 打印结果
    print(f"列表的总和是: {total}")
    ```

2.  按下 `Shift + Enter` 运行它。你会看到输出：`列表的总和是: 160`。

### 4. 可视化初体验

数据分析离不开可视化。让我们用 `matplotlib` 库来画一个简单的图。

1.  在一个新的 Code 单元格中，输入以下代码：

    ```python
    # 导入 matplotlib 库
    import matplotlib.pyplot as plt

    # 准备数据
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 3, 4] 

    # 创建一个简单的折线图
    plt.plot(x, y)

    # 添加标题和标签
    plt.title("My First Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 显示图表
    plt.show()
    ```

2.  按下 `Shift + Enter` 运行。你会看到一张漂亮的折线图直接内嵌在你的 Notebook 中！

    ![Notebook 中生成的 matplotlib 折线图]({{ site.baseurl}}/assets/img_ana/Section6_jupyter_first_code1.png)
    
    *图：Notebook 中生成的 matplotlib 折线图（示例）*

## 重要的快捷键（命令模式）

熟练使用快捷键是提升效率的关键。

*   `Shift + Enter`: 运行当前单元格，并自动选择下一个单元格。
*   `Ctrl + Enter` (或 `Cmd + Enter`): 运行当前单元格，但停留在当前单元格。
*   `A`: 在当前单元格**上方 (Above)** 插入一个新单元格。
*   `B`: 在当前单元格**下方 (Below)** 插入一个新单元格。
*   `M`: 将当前单元格切换为 **Markdown** 模式。
*   `Y`: 将当前单元格切换为 **Code** 模式。
*   `D, D` (按两次 D): 删除当前单元格。
*   `Z`: 撤销删除单元格。

## 保存和关闭

*   **保存：** Jupyter 会自动保存你的工作。你也可以手动按 `Ctrl + S` (或 `Cmd + S`) 或者点击工具栏上的保存图标来保存。
*   **关闭：** 关闭浏览器标签页即可。然后回到你启动 Jupyter 的那个命令行终端，按 `Ctrl + C` 两次，即可停止 Jupyter 服务。

恭喜你！你已经完成了 Jupyter Notebook 的基本操作。现在，你可以尝试修改代码、添加新的笔记，真正开始你的数据探索之旅了。
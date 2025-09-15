---
title: 3. 安装指南：一步步带你配置好环境
author: Anda Li
date: 2024-01-03
category: Data Science Learning
layout: post
---

在了解了基本概念之后，现在让我们动手开始安装 Anaconda。这个过程非常简单，只需要跟随下面的步骤即可。

## 下载 Anaconda

首先，我们需要从 Anaconda 的官方网站下载安装包。 

1.  **访问官网：** 打开浏览器，访问 Anaconda 的官方下载页面：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
   
2.  **选择操作系统：** 网站会自动检测你的操作系统（Windows, macOS, or Linux），并推荐相应的下载版本。你只需要点击下载按钮即可。

    ![Anaconda 官网下载页面，箭头指向下载按钮]({{ site.baseurl}}/assets/img_ana/4_downloadAnaconda.png)
    
    *该截图展示了 Anaconda 官方下载页面，并用箭头标注了推荐的下载按钮。若页面未正确识别你的操作系统，可手动选择 Windows / macOS / Linux 后再点击下载。*

> Anaconda 的安装包比较大（通常在 500MB 以上），下载可能需要一些时间。
> 
> **备选建议:** 也可访问清华镜像（TUNA）下载 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/), 注意选择正确版本。如果你是Windows系统，可直接点击入选链接下载:[Anaconda3-2025.06-1-Windows-x86_64.exe](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2025.06-1-Windows-x86_64.exe) 。
{: .block-tip }

## 安装 Anaconda (以 Windows 为例)

下载完成后，找到安装包（通常在你的“下载”文件夹中），双击开始安装。macOS 和 Linux 的安装过程与此类似。

1.  **开始安装：** 双击 `.exe` 安装文件，你会看到欢迎界面。点击 **“Next”** 继续。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_1.png)

    *图：Anaconda 安装欢迎界面*

2.  **许可协议：** 阅读许可协议，然后点击 **“I Agree”**。

3.  **安装类型：** 选择为“Just Me”还是“All Users”。通常情况下，选择 **“Just Me”** 即可，这样会将 Anaconda 安装在你的用户目录下，避免权限问题。然后点击 **“Next”**。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_2.png)

    *图：安装类型选择界面（推荐选择 “Just Me”）*

4.  **选择安装路径：** 这一步非常重要！

    *   **建议：** 保持默认的安装路径。默认路径通常是 `C:\Users\YourUsername\anaconda3`。

    > **不要**将 Anaconda 安装在包含**空格**或**中文字符**的路径下！例如，`C:\Program Files` 或 `C:\Users\张三\Desktop` 这样的路径可能会导致未知的错误。如果你想自定义路径，请确保路径只包含英文字母和数字。
    {: .block-warning }


    * 选择好路径后，点击 **“Next”**。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_3.png)
    
    *图：选择安装路径（请避免空格与中文字符）*

1.  **高级选项：** 这是最关键的一步，请务必注意！
    
    *   你会看到两个选项：
        1.  `Add Anaconda3 to my PATH environment variable` (将 Anaconda3 添加到我的 PATH 环境变量中)
        2.  `Register Anaconda3 as my default Python 3.x` (将 Anaconda3 注册为我的默认 Python 3.x)

    *   **官方建议：** 安装程序会提示**不建议**勾选第一个选项（Add to PATH），因为这可能会干扰其他软件。他们的建议是使用 “Anaconda Prompt” 来访问 anaconda。
    *   **我们的建议（更适合初学者）：** **直接勾选第一个选项！** 尽管官方不推荐，但对于初学者来说，勾选此项可以让你在系统的任何命令行终端（如 Windows 的 `cmd` 或 `PowerShell`）中直接使用 `conda`、`python`、`pip` 等命令，这会极大地简化你后续的操作，避免很多不必要的麻烦。第二个选项保持默认勾选即可。
    *   **总结：** **两个选项都勾选！** 然后点击 **“Install”** 开始安装。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_4.png)

    *图：高级选项（建议两个选项都勾选以便初学者使用）*

2.  **安装过程：** 安装过程会持续几分钟，请耐心等待。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_5.png)

    *图：安装进度界面*

3.  **安装完成：** 安装完成后，点击 **“Next”**。

4.  **完成：** 最后，你会看到一个感谢页面，取消勾选 “Launch Anaconda Navigator” 和 “Getting started with Anaconda”，然后点击 **“Finish”** 即可。

    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_6.png)

    *图：安装完成界面*

## 验证安装

如何确认 Anaconda 是否已经成功安装了呢？

1.  **打开命令行终端：**
    *   **Windows:** 按下 `Win` 键，输入 `cmd`，然后按回车，打开命令提示符。
    *   **macOS:** 按下 `Command + Space`，输入 `Terminal`，然后按回车，打开终端。

2.  **输入命令：** 在打开的命令行窗口中，输入以下命令，然后按回车：

    ```bash
    conda --version
    ```

3.  **查看输出：** 如果安装成功，它会显示出 `conda` 的版本号，例如 `conda 23.7.4`。   

    接着，你可以用同样的方式验证 `python`：

    ```bash
    python --version
    ```

    如果显示出 Python 的版本号（例如 `Python 3.11.5`），那么恭喜你，你的 Anaconda 环境已经完全准备好了！

    ![查看版本]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_7_cmd.png)

    *图：查看版本*

## 启动 Jupyter Notebook

现在，让我们来启动我们的交互式笔记本 Jupyter Notebook。

1.  **打开命令行终端：** 同样，打开 `cmd` 或 `Terminal`。

2.  **输入命令：** 在命令行中输入以下命令，然后按回车：

    ```bash
    jupyter notebook
    ```
    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_8_jupyter_notebook.png)
    *图：运行 jupyter notebook*


3.  **自动打开浏览器：** 执行命令后，你的默认浏览器会自动打开一个新标签页，显示 Jupyter Notebook 的主界面。这个界面展示的是你执行命令时所在的目录下的文件。
   
    ![]({{ site.baseurl}}/assets/img_ana/5_Anaconda_in_8_jupyter_notebook_browser.png)

    *图：Jupyter Notebook 在浏览器中启动后的主界面截图*


> 只要 Jupyter Notebook 正在运行，**不要关闭**那个你用来启动它的命令行窗口！关闭它会导致 Jupyter Notebook 服务中断。
{: .block-danger }

## 参考视频教程

如果你更倾向于视频教学，可以参考这段安装演示（含下载、安装选项、PATH 设置、以及启动 Jupyter 的完整流程）：
https://www.bilibili.com/video/BV1y6XGYUEP7/?spm_id_from=333.337.search-card.all.click


## 结语
至此，你已经完成了从下载、安装到启动的全过程。在下一章节，我们将学习如何更有效地管理我们的 Anaconda 环境。
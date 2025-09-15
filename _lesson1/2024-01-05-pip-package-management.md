---
title: 5. PIP 和包管理：为你的环境“添砖加瓦”
author: Anda Li
date: 2024-01-05
category: Data Science Learning
layout: post
---

在上一章，我们学习了如何使用 `conda` 来管理我们的环境和包。`conda` 是 Anaconda 生态系统的核心，但它并不是 Python 世界里唯一的包管理器。另一个你必须了解的工具，就是 `pip`。

## 什么是 pip？

*   **pip** 是 “Pip Installs Packages” 的递归缩写。它是 Python 官方推荐的包管理器，也是最通用的 Python 包安装和管理工具。
*   当你安装 Python 时（无论是通过官网的安装包还是通过 Anaconda），`pip` 通常都会被自动安装。
*   `pip` 从一个名为 PyPI (Python Package Index) 的在线仓库中下载和安装包。PyPI 是 Python 官方的、也是最大的第三方软件包仓库，几乎所有开源的 Python 库都会在这里发布。

## `conda` vs `pip`：我应该用哪个？

初学者常常会对 `conda` 和 `pip` 的关系感到困惑。下面这个表格可以帮助你清晰地理解它们的区别：

<div class="table-wrapper" markdown="block">

| 特性         | `conda`                                        | `pip`                                                |
| ------------ | ---------------------------------------------- | ---------------------------------------------------- |
| **来源**     | Anaconda, Inc.                                 | Python 官方 (PyPA)                                   |
| **功能**     | **环境管理器 + 包管理器**                      | **仅包管理器**                                       |
| **包的语言** | 管理 **任何语言** 的包 (Python, R, C++, etc.)  | 只管理 **Python** 包                                 |
| **仓库**     | Anaconda Repository (channels)                 | PyPI (Python Package Index)                          |
| **环境管理** | 可以创建、激活、删除独立的环境                 | 无法管理环境（需要依赖 `venv` 或 `virtualenv` 等工具） |
| **依赖解析** | 依赖解析能力更强，特别是对非 Python 库的依赖 | 可能会在处理复杂的依赖关系时遇到问题                 |

</div>

**核心思想：**

1.  **环境管理用 `conda`：** 创建、激活、删除环境，这些都应该使用 `conda` (`conda create`, `conda activate`, `conda deactivate`)。`pip` 没有这个能力。
2.  **包安装首选 `conda`：** 在一个激活的 Conda 环境中，当你需要安装一个新的包时，应该 **首先尝试 `conda install <package_name>`**。因为 `conda` 会更好地处理包与包之间的依赖关系，确保整个环境的稳定性。
3.  **`conda` 找不到时用 `pip`：** Anaconda 的仓库虽然庞大，但有时可能没有你需要的某个特定的、或者非常新的包。如果在 `conda` 中找不到（`conda install` 失败），那么 `pip` 就是你的第二选择。在 **同一个激活的环境** 中，直接运行 `pip install <package_name>`。

**总结一句话：** 用 `conda` 来管理环境，用 `conda` 来安装大部分的包，用 `pip` 作为 `conda` 的补充。

## pip 的常用命令

假设你已经激活了一个 Conda 环境（例如 `conda activate my_project`），下面是 `pip` 的一些常用命令。

*   **安装包：**

    ```bash
    pip install requests
    ```

*   **安装特定版本的包：**

    ```bash
    pip install pandas==1.2.5
    ```

*   **升级包：**

    ```bash
    pip install --upgrade numpy
    ```

*   **卸载包：**

    ```bash
    pip uninstall requests
    ```

*   **查看已安装的包：**

    ```bash
    pip list
    ```

*   **查看某个包的详细信息：**

    ```bash
    pip show pandas
    ```

> requests, pandas, numpy 等是Python中的包，根据你的需要安装包。默认Anaconda已经包含了很多常用的数据分析包，当你运行所需的包不存在时，可以使用 pip install 进行安装。
{: .block-tip }

## 配置国内镜像源：让下载“飞起来”

无论是 `conda` 还是 `pip`，它们的默认服务器都在国外。由于网络原因，我们直接从官方源下载包时，速度可能会非常慢，甚至下载失败。 

为了解决这个问题，我们可以将下载源配置为国内的镜像服务器。这些镜像服务器会定期从官方同步，保证我们能快速地访问到最新的资源。国内有很多优秀的镜像源，比如清华大学的 TUNA 镜像、阿里云镜像、豆瓣镜像等。

### 配置 pip 镜像源

配置 `pip` 的镜像源非常简单，只需要一个命令即可。我们以清华 TUNA 镜像为例（官方地址： https://mirrors.tuna.tsinghua.edu.cn/ 或者 https://pypi.tuna.tsinghua.edu.cn/）。

![清华 TUNA 镜像配置示意图]({{ site.baseurl}}/assets/img_ana/Section5_TUNA.png)

 *图：清华 TUNA 镜像配置示意图*

```bash
# 临时使用
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package_name>

# 永久配置 (推荐！)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

执行第二条命令后，`pip` 就会永久地将下载源设置为清华镜像。以后你再使用 `pip install` 时，就会自动从国内镜像下载，速度会得到质的飞跃。

### 配置 conda 镜像源

同样，我们也可以为 `conda` 配置镜像源。`conda` 的包是通过 “channels” 来管理的。我们可以将清华的镜像 channel 添加到 `conda` 的配置中。

打开命令行终端，依次执行以下命令：

```bash
# 添加清华大学的 anaconda 镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

# 设置搜索时显示 channel 的 URL，方便我们确认镜像源是否生效
conda config --set show_channel_urls yes
```

配置完成后，你可以通过 `conda install` 来测试一下。在安装包时，你会看到显示的下载链接已经是 `mirrors.tuna.tsinghua.edu.cn` 开头的了，这说明你的配置已经生效。

> 在开始使用 Anaconda 和 pip 之后，**第一时间**就把 `conda` 和 `pip` 的镜像源都配置好。这将为你节省大量等待下载的时间。
{: .block-tip }
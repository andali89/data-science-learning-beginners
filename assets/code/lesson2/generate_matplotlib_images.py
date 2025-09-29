"""Generate Matplotlib example figures for the Matplotlib quickstart chapter."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


def configure_fonts() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_figure_axes_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Figure boundary
    ax.add_patch(
        patches.Rectangle(
            (0.5, 0.5),
            9,
            5,
            facecolor="#f2f2f2",
            edgecolor="#333333",
            linewidth=2,
            zorder=1,
        )
    )
    ax.text(5, 5.4, "Figure (画布)", ha="center", va="center", fontsize=14, weight="bold")

    # Axes examples
    axes_specs = [
        (1.5, 3.5, "Axes 1\n坐标系"),
        (5.0, 2.0, "Axes 2"),
        (7.0, 3.8, "Axes 3"),
    ]
    for x, y, label in axes_specs:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                2.5,
                1.4,
                facecolor="#d7ebff",
                edgecolor="#1f77b4",
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(x + 1.25, y + 0.7, label, ha="center", va="center", fontsize=12)

    ax.annotate(
        "Figure 可以包含多个 Axes",
        xy=(5.0, 1.2),
        xytext=(5.0, 0.3),
        ha="center",
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        fontsize=11,
    )

    fig.savefig(output_dir / "figure_4_1_figure_axes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_line_plot(output_dir: Path) -> None:
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        x_data,
        y_data,
        color="steelblue",
        linewidth=2,
        marker="o",
        markevery=10,
        label="sin(x)",
    )
    ax.set_title("简单的正弦函数折线图")
    ax.set_xlabel("X 值")
    ax.set_ylabel("sin(X)")
    ax.legend(loc="best")

    fig.savefig(output_dir / "figure_4_2_line_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_bar_plots(output_dir: Path) -> None:
    categories = ["第一季度", "第二季度", "第三季度", "第四季度"]
    values = [150, 230, 180, 210]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    ax1.bar(
        categories,
        values,
        width=0.6,
        color=colors,
        edgecolor="black",
        label="季度销售",
    )
    ax1.set_title("垂直柱状图 (bar)")
    ax1.set_ylabel("销售额 (万元)")
    ax1.legend()

    ax2.barh(
        categories,
        values,
        height=0.6,
        color=colors,
        label="季度销售",
        alpha=0.8,
    )
    ax2.set_title("水平柱状图 (barh)")
    ax2.set_xlabel("销售额 (万元)")
    ax2.legend(loc="lower right")

    fig.suptitle("年度销售额对比")

    fig.savefig(output_dir / "figure_4_3_bar_charts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_histogram(output_dir: Path, rng: np.random.Generator) -> None:
    data = rng.normal(loc=0, scale=1, size=1000)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        data,
        bins=30,
        range=(-4, 4),
        color="#66b3ff",
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_title("随机数据的分布直方图")
    ax.set_xlabel("数值")
    ax.set_ylabel("频数")

    fig.savefig(output_dir / "figure_4_4_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_scatter(output_dir: Path, rng: np.random.Generator) -> None:
    x_scatter = rng.uniform(0, 10, size=50)
    y_scatter = 2 * x_scatter + 1 + rng.normal(scale=2, size=50)
    sizes = (rng.random(50) * 80) + 20

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        x_scatter,
        y_scatter,
        s=sizes,
        c=x_scatter,
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        label="样本点",
    )
    ax.set_title("X 和 Y 的散点关系图")
    ax.set_xlabel("变量 X")
    ax.set_ylabel("变量 Y")
    ax.grid(True)
    ax.legend(loc="upper left")

    fig.savefig(output_dir / "figure_4_5_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_pie_chart(output_dir: Path) -> None:
    labels = ["市场部", "研发部", "销售部", "行政部"]
    sizes = [15, 30, 45, 10]
    explode = (0, 0, 0.1, 0)
    colors = ["#ff9999", "#66b3ff", "#ffcc99", "#99ff99"]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        pctdistance=0.8,
    )
    ax.axis("equal")
    ax.set_title("公司各部门人数占比")

    fig.savefig(output_dir / "figure_4_6_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_boxplot(output_dir: Path, rng: np.random.Generator) -> None:
    data1 = rng.normal(100, 10, 200)
    data2 = rng.normal(80, 30, 200)
    data3 = rng.normal(90, 20, 200)
    data_to_plot = [data1, data2, data3]

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        labels=["A产品", "B产品", "C产品"],
        showfliers=True,
        widths=0.6,
    )

    colors = ["#99ff99", "#66b3ff", "#ff9999"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_title("不同产品销售额分布对比")
    ax.set_ylabel("销售额")
    ax.yaxis.grid(True)

    fig.savefig(output_dir / "figure_4_7_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_title_label_legend(output_dir: Path) -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="sin(x)")
    ax.plot(x, y2, label="cos(x)", linestyle="--")
    ax.set_title("正弦与余弦函数图像")
    ax.set_xlabel("X 轴 (弧度)")
    ax.set_ylabel("Y 轴 (值)")
    ax.legend()

    fig.savefig(output_dir / "figure_4_8_labels_legend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_custom_style(output_dir: Path) -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="sin(x)", color="blue", linestyle="-", marker="o", markersize=2)
    ax.plot(x, y2, label="cos(x)", color="red", linestyle=":")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("定制化样式的图表")
    ax.legend()

    fig.savefig(output_dir / "figure_4_9_custom_style.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_bar_with_annotations(output_dir: Path) -> None:
    categories = ["非常不满意", "不满意", "一般", "满意", "非常满意"]
    values = [5, 25, 50, 120, 200]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color="#66b3ff")
    ax.tick_params(axis="x", rotation=45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{int(yval)}",
            va="bottom",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("客户满意度调查结果")
    ax.set_ylabel("投票数")
    fig.tight_layout()

    fig.savefig(output_dir / "figure_4_10_bar_annotations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_subplots_grid(output_dir: Path, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        constrained_layout=True,
        squeeze=True,
    )

    axes[0, 0].plot(np.sin(np.linspace(0, 10, 100)))
    axes[0, 0].set_title("折线图")

    axes[0, 1].bar(["A", "B", "C"], [3, 5, 2])
    axes[0, 1].set_title("柱状图")

    axes[1, 0].hist(rng.normal(size=500), bins=20)
    axes[1, 0].set_title("直方图")

    axes[1, 1].scatter(rng.uniform(size=50), rng.uniform(size=50))
    axes[1, 1].set_title("散点图")

    fig.savefig(output_dir / "figure_4_11_subplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_savefig_example(output_dir: Path) -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y1, label="sin(x)")
    ax.set_title("保存这张图")
    ax.legend()

    fig.savefig(
        output_dir / "figure_4_12_saved_plot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def main() -> None:
    output_dir = Path("images/matplotlib")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    rng = np.random.default_rng(42)

    draw_figure_axes_diagram(output_dir)
    generate_line_plot(output_dir)
    generate_bar_plots(output_dir)
    generate_histogram(output_dir, rng)
    generate_scatter(output_dir, rng)
    generate_pie_chart(output_dir)
    generate_boxplot(output_dir, rng)
    generate_title_label_legend(output_dir)
    generate_custom_style(output_dir)
    generate_bar_with_annotations(output_dir)
    generate_subplots_grid(output_dir, rng)
    generate_savefig_example(output_dir)

    # produce an example DataFrame line plot for potential future use
    sales = pd.DataFrame(
        {
            "季度": ["Q1", "Q2", "Q3", "Q4"],
            "销售额": [150, 230, 180, 210],
        }
    )
    fig, ax = plt.subplots()
    ax.plot(sales["季度"], sales["销售额"], marker="o")
    ax.set_title("季度销售趋势")
    ax.set_ylabel("销售额 (万元)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_extra_quarterly_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

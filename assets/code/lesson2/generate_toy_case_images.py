"""Generate figures for the toy sales analysis case study."""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.unicode_minus"] = False

    preferred_fonts = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            break


def prepare_sales_dataframe() -> pd.DataFrame:
    data = {
        "OrderID": [
            "O001",
            "O002",
            "O003",
            "O004",
            "O005",
            "O006",
            "O007",
            "O008",
            "O009",
            "O010",
            "O011",
            "O012",
            "O013",
            "O014",
            "O014",
        ],
        "Date": [
            "2023-01-15",
            "2023-01-20",
            "2023-02-05",
            "2023-02-12",
            "2023-02-21",
            "2023-03-04",
            "2023-03-10",
            "2023-03-18",
            "2023-03-25",
            "2023-01-28",
            "2023-02-15",
            "2023-03-22",
            "2023-01-25",
            "2023-03-30",
            "2023-03-30",
        ],
        "Category": [
            "电子产品",
            "家居用品",
            "电子产品",
            "图书",
            "家居用品",
            "电子产品",
            "图书",
            "电子产品",
            "家居用品",
            "图书",
            "图书",
            "家居用品",
            "电子产品",
            "电子产品",
            "电子产品",
        ],
        "Product": [
            "笔记本电脑",
            "咖啡机",
            "无线鼠标",
            "数据分析入门",
            "智能台灯",
            "键盘",
            "Python编程",
            "显示器",
            "储物盒",
            "机器学习实战",
            "Web开发",
            "香薰机",
            "游戏手柄",
            "充电宝",
            "充电宝",
        ],
        "Price": [
            7000,
            800,
            150,
            60,
            200,
            450,
            70,
            2000,
            80,
            80,
            55,
            150,
            300,
            120,
            120,
        ],
        "Quantity": [1, 2, 5, 10, 3, 2, 8, 1, 15, 6, 9, 2, 2, 1, 1],
    }

    sales_df = pd.DataFrame(data)
    sales_df = sales_df.drop_duplicates().reset_index(drop=True)
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    sales_df["Month"] = sales_df["Date"].dt.month
    sales_df["TotalSale"] = sales_df["Price"] * sales_df["Quantity"]
    return sales_df


def compute_aggregations(sales_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    monthly_sales = sales_df.groupby("Month", as_index=False)["TotalSale"].sum()
    month_map = {1: "一月", 2: "二月", 3: "三月"}
    monthly_sales["MonthName"] = monthly_sales["Month"].map(month_map)
    monthly_sales = monthly_sales.sort_values("Month").reset_index(drop=True)

    category_analysis = (
        sales_df.groupby("Category")
        .agg(
            TotalRevenue=("TotalSale", "sum"),
            TotalQuantity=("Quantity", "sum"),
            OrderCount=("OrderID", "nunique"),
        )
        .sort_values(by="TotalRevenue", ascending=False)
    )

    product_sales = sales_df.groupby("Product")["TotalSale"].sum().sort_values(ascending=False)
    top_3_products = product_sales.head(3)

    return monthly_sales, category_analysis, top_3_products


def generate_dashboard(
    output_dir: Path,
    monthly_sales: pd.DataFrame,
    category_analysis: pd.DataFrame,
    top_3_products: pd.Series,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(
        monthly_sales["MonthName"],
        monthly_sales["TotalSale"],
        marker="o",
        linestyle="-",
        color="#1f77b4",
    )
    axes[0].set_title("1. 月度销售趋势", fontsize=14)
    axes[0].set_ylabel("总销售额 (元)")

    bars = axes[1].bar(
        category_analysis.index,
        category_analysis["TotalRevenue"],
        color=["#ff7f0e", "#2ca02c", "#d62728"],
    )
    axes[1].set_title("2. 各类别销售额对比", fontsize=14)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{int(yval)}",
            va="bottom",
            ha="center",
        )

    axes[2].barh(
        top_3_products.index,
        top_3_products.values,
        color=["#9467bd", "#8c564b", "#e377c2"],
    )
    axes[2].set_title("3. Top 3 畅销产品", fontsize=14)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("总销售额 (元)")

    fig.suptitle("第一季度销售分析报告", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / "figure_5_1_sales_dashboard.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path("images/toy-case")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    sales_df = prepare_sales_dataframe()
    monthly_sales, category_analysis, top_3_products = compute_aggregations(sales_df)
    generate_dashboard(output_dir, monthly_sales, category_analysis, top_3_products)


if __name__ == "__main__":
    main()

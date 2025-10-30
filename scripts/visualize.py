#!/usr/bin/env python3
"""
可视化脚本：根据 data.xlsx 输出核心描述性图表。

默认生成：
1. 机器人渗透率 vs 价值链地位散点图（按年份着色）。
2. 渗透率与价值链地位随时间演化的折线图（选定国家）。
3. 候选关键变量与 y 的相关热力图。

示例：
    python scripts/visualize.py --countries China,Germany,'United States' --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robot_analysis.cli import (
    apply_transformations,
    load_dataset,
    prepare_panel,
)

sns.set_theme(style="whitegrid", font="SimHei", rc={"axes.unicode_minus": False})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成机器人渗透率与价值链地位可视化图表")
    parser.add_argument("--data", type=Path, default=Path("data/data.xlsx"), help="数据文件路径")
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Excel 工作表名")
    parser.add_argument("--countries", type=str, default="", help="指定需要重点展示的国家，逗号分隔")
    parser.add_argument(
        "--top-countries",
        type=int,
        default=5,
        help="若未指定国家，则选取机器人渗透率均值前 N 的国家用于折线图",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="图表输出目录",
    )
    parser.add_argument(
        "--corr-variables",
        type=str,
        default="",
        help="自定义热力图变量列表（原始列名或逗号分隔），默认自动选择与 y 相关性最高的变量",
    )
    parser.add_argument(
        "--max-corr-vars",
        type=int,
        default=10,
        help="自动选择相关变量数量上限",
    )
    parser.add_argument("--dpi", type=int, default=200, help="图像分辨率 (dpi)")
    return parser.parse_args(argv)


def select_countries(panel: pd.DataFrame, explicit: Sequence[str], top_n: int) -> List[str]:
    if explicit:
        return [c for c in explicit if c in panel["country"].unique()]
    top_countries = (
        panel.groupby("country")["x"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return top_countries


def auto_select_corr_vars(panel: pd.DataFrame, max_vars: int) -> List[str]:
    numeric_cols = panel.select_dtypes(include="number").columns.drop(["y", "x"], errors="ignore")
    correlations = {}
    for col in numeric_cols:
        subset = panel[["y", col]].dropna()
        if len(subset) < 50:
            continue
        correlations[col] = subset["y"].corr(subset[col])
    top = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_vars]
    return [col for col, _ in top]


def plot_scatter(panel: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=panel,
        x="x",
        y="y",
        hue="year",
        palette="viridis",
        ax=ax,
        alpha=0.7,
    )
    ax.set_xlabel("机器人渗透率 (x)")
    ax.set_ylabel("全球价值链地位 (y)")
    ax.set_title("机器人渗透率与价值链地位关系")
    ax.legend(title="年份", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "scatter_x_y.png", dpi=dpi)
    plt.close(fig)


def plot_time_series(panel: pd.DataFrame, countries: Sequence[str], output: Path, dpi: int) -> None:
    subset = panel[panel["country"].isin(countries)]
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    sns.lineplot(data=subset, x="year", y="x", hue="country", marker="o", ax=axes[0])
    axes[0].set_ylabel("机器人渗透率 (x)")
    axes[0].set_title("机器人渗透率演化")
    sns.lineplot(data=subset, x="year", y="y", hue="country", marker="o", ax=axes[1])
    axes[1].set_ylabel("价值链地位 (y)")
    axes[1].set_title("价值链地位演化")
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "timeseries_x_y.png", dpi=dpi)
    plt.close(fig)


def plot_corr_heatmap(panel: pd.DataFrame, variables: Sequence[str], output: Path, dpi: int) -> None:
    cols = ["y"] + list(variables)
    data = panel[cols].dropna()
    if data.empty:
        return
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(cols), 0.8 * len(cols)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("与价值链地位相关的变量热力图")
    fig.tight_layout()
    fig.savefig(output / "heatmap_corr.png", dpi=dpi)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    df_raw, mapping = load_dataset(args.data, args.sheet)
    panel = prepare_panel(df_raw, mapping)
    panel = apply_transformations(panel, log_cols=[], winsorize=True, add_log1p_for_x=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    explicit_countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    focus_countries = select_countries(panel, explicit_countries, args.top_countries)
    print(f"折线图展示国家：{', '.join(focus_countries)}")

    if args.corr_variables:
        corr_vars = [mapping.get(var.strip(), var.strip()) for var in args.corr_variables.split(",")]
    else:
        corr_vars = auto_select_corr_vars(panel, args.max_corr_vars)

    plot_scatter(panel, args.output_dir, args.dpi)
    plot_time_series(panel, focus_countries, args.output_dir, args.dpi)
    plot_corr_heatmap(panel, corr_vars, args.output_dir, args.dpi)
    print(f"图表已保存至 {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

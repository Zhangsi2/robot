#!/usr/bin/env python3
"""
辅助分析脚本：基于项目数据集的变量组合搜索与拟合优度评估。

主要功能：
1. 数据读取与列名清洗，保留 y/x/country/year 面板结构。
2. 根据预定义或自定义的候选变量集合，执行前向逐步回归或子集枚举，寻找拟合度较好的变量组合。
3. 对每个模型输出 R^2、调整后 R^2、AIC、BIC、RMSE、x 系数及显著性等指标，可保存为 CSV。

用法示例：
    python -m robot_analysis --method stepwise --groups macro,industry,innovation --max-vars 6 --output results_stepwise.csv
    python -m robot_analysis --method subsets --groups trade,finance --max-vars 4 --max-combos 200 --standardize

依赖库：pandas, numpy, statsmodels。若未安装，请先执行 pip install pandas numpy statsmodels openpyxl。
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 默认数据路径指向项目根目录下的 data/data.xlsx，若路径不存在则需显式指定。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data.xlsx"
_SMF = None  # 缓存 statsmodels.formula.api

# === 配置：候选变量分组（使用原始列名） ============================================

CANDIDATE_GROUPS: Dict[str, List[str]] = {
    "macro": [
        "GDP（现价美元）",
        "人均 GDP（现价美元）",
        "人均 GDP增长（年增长率）",
        "固定资本形成总额（占 GDP 的百分比）",
        "资本形成总额（占 GDP 的百分比）",
    ],
    "industry": [
        "制造业，增加值（占GDP的百分比）",
        "制造业出口（占商品出口的百分比）",
        "工业增加值（占 GDP 的百分比）",
        "矿石和金属出口（占商品出口的百分比）",
    ],
    "innovation": [
        "研发支出（占GDP的比例）",
        "R&D研究人员 （每百万人）",
        "每100万人中研发技术人员的数量",
        "高科技出口（占制成品出口的百分比）",
        "科技期刊文章",
    ],
    "trade": [
        "货物和服务出口（占 GDP 的百分比）",
        "货物和服务进口（占 GDP 的百分比）",
        "服务贸易额（占国民生产总值（GDP）比例）",
        "商品贸易（GDP的百分比）",
        "国际旅游，支出（占总进口的百分比）",
        "国际旅游，收入（占总出口的百分比）",
    ],
    "finance": [
        "广义货币（占 GDP 的百分比）",
        "私营部门的国内信贷（占 GDP 的百分比）",
        "股票交易总额（占国民生产总值（GDP）的比例）",
        "自动取款机（ATM）(每10万成年人)",
        "商业银行分支机构 (每10万成年人)",
    ],
    "infrastructure": [
        "物流绩效指数：综合分数（1=很低 至 5=很高）",
        "安全互联网服务器（每百万人）",
        "Mobile cellular subscriptions (per 100 people)",
    ],
    "human_capital": [
        "入学率，高等院校（占总人数的百分比）",
        "公共教育支出，总数（占政府支出的比例）",
        "人均居民最终消费支出（年增长率）",
        "劳动力参与率，总数（占 15 岁以上总人口的百分比）（模拟劳工组织估计）",
    ],
}


# === 数据预处理工具函数 =============================================================

def sanitize_columns(columns: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    """将列名转换为安全的 snake_case，用于 statsmodels 公式。

    返回 (新列名列表, 原始 -> 新列名映射)。
    """
    mapping: Dict[str, str] = {}
    counters: Dict[str, int] = {}
    used: set[str] = set()
    sanitized: List[str] = []

    for col in columns:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", col)
        safe = re.sub(r"_+", "_", safe).strip("_").lower()
        if not safe or safe[0].isdigit():
            safe = f"col_{abs(hash(col)) % 10**6}"
        base = safe
        while safe in used:
            counters[base] = counters.get(base, 0) + 1
            safe = f"{base}_{counters[base]}"
        counters.setdefault(base, 0)
        used.add(safe)
        mapping[col] = safe
        sanitized.append(safe)
    return sanitized, mapping


def load_dataset(path: Path, sheet: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """读取 Excel 并执行列名清洗，返回清洗后的 DataFrame 与映射。"""
    df_raw = pd.read_excel(path, sheet_name=sheet)
    sanitized_cols, mapping = sanitize_columns(df_raw.columns)
    df = df_raw.copy()
    df.columns = sanitized_cols
    return df, mapping


def resolve_columns(names: Sequence[str], mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """将原始列名映射到清洗后的列名，返回 (存在列表, 缺失列表)。"""
    resolved: List[str] = []
    missing: List[str] = []
    for name in names:
        if name not in mapping:
            missing.append(name)
        else:
            resolved.append(mapping[name])
    return resolved, missing


def prepare_panel(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """统一核心列名，并过滤缺失的基本信息。"""
    required = ["y", "x", "country", "year"]
    rename_map = {
        mapping.get("y", "y"): "y",
        mapping.get("x", "x"): "x",
        mapping.get("country", "country"): "country",
        mapping.get("year", "year"): "year",
    }
    df = df.rename(columns=rename_map)
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise KeyError(f"缺少核心变量列：{missing_required}")

    panel = df[["year", "country", "y", "x"]].copy()
    panel["country"] = panel["country"].astype(str)
    panel["year"] = panel["year"].astype(int)
    # 合并其他列
    for col in df.columns:
        if col not in panel.columns:
            panel[col] = df[col]
    # 去除 y/x/country/year 缺失的观测
    panel = panel.dropna(subset=["y", "x", "country", "year"])
    return panel


# === 模型评估工具函数 ===============================================================

def standardize_columns(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """对指定列进行标准化处理（均值 0、方差 1）。"""
    data = data.copy()
    for col in columns:
        std = data[col].std(ddof=0)
        if std is None or math.isclose(std, 0.0):
            raise ValueError(f"变量 {col} 方差为 0，无法标准化。")
        data[col] = (data[col] - data[col].mean()) / std
    return data


def evaluate_model(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    standardize: bool = False,
    min_obs: int = 60,
) -> Optional[Dict[str, object]]:
    """在包含国家/年份固定效应的框架下拟合模型并返回指标。"""
    subset_cols = ["y", "x", "country", "year"] + list(features)
    data = df[subset_cols].dropna()
    if len(data) < min_obs or data["country"].nunique() < 2 or data["year"].nunique() < 2:
        return None
    if standardize and features:
        try:
            data = standardize_columns(data, features)
        except ValueError:
            return None

    formula = "y ~ x"
    if features:
        feature_terms = " + ".join(features)
        formula += " + " + feature_terms
    formula += " + C(country) + C(year)"

    global _SMF
    if _SMF is None:
        try:
            import statsmodels.formula.api as smf  # type: ignore
        except ImportError as exc:  # pragma: no cover - 运行环境缺失依赖时给出提示
            raise SystemExit(
                "缺少依赖库 statsmodels，请先运行 `pip install statsmodels` 或参考 README 安装依赖后再执行脚本。"
            ) from exc
        _SMF = smf
    else:
        smf = _SMF  # type: ignore[assignment]

    try:
        model = smf.ols(formula=formula, data=data)
        result = model.fit()
    except Exception:
        return None

    residuals = result.resid
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    metrics: Dict[str, object] = {
        "n_obs": int(len(data)),
        "r2": float(result.rsquared),
        "adj_r2": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "rmse": rmse,
        "coef_x": float(result.params.get("x", np.nan)),
        "pvalue_x": float(result.pvalues.get("x", np.nan)),
        "features": list(features),
    }
    return metrics


def forward_stepwise(
    df: pd.DataFrame,
    candidates: Sequence[str],
    base_vars: Sequence[str],
    *,
    max_vars: int,
    min_improvement: float,
    standardize: bool,
) -> List[Dict[str, object]]:
    """基于调整后 R^2 的前向逐步回归。"""
    selected = list(base_vars)
    history: List[Dict[str, object]] = []
    current_score: Optional[float] = None

    remaining = [var for var in candidates if var not in selected]

    while remaining and len(selected) < max_vars:
        best_candidate: Optional[str] = None
        best_metrics: Optional[Dict[str, object]] = None

        for candidate in remaining:
            trial_features = selected + [candidate]
            metrics = evaluate_model(df, trial_features, standardize=standardize)
            if not metrics:
                continue
            if best_metrics is None or metrics["adj_r2"] > best_metrics["adj_r2"]:
                best_candidate = candidate
                best_metrics = metrics

        if best_candidate is None or best_metrics is None:
            break

        improvement = (
            float("inf") if current_score is None else best_metrics["adj_r2"] - current_score
        )
        if current_score is not None and improvement < min_improvement:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_score = best_metrics["adj_r2"]

        step_info = best_metrics.copy()
        step_info["selected_features"] = list(selected)
        history.append(step_info)

    return history


def enumerate_subsets(
    df: pd.DataFrame,
    candidates: Sequence[str],
    base_vars: Sequence[str],
    *,
    max_vars: int,
    max_combos: Optional[int],
    standardize: bool,
) -> List[Dict[str, object]]:
    """穷举（或限定数量）变量组合，按照调整后 R^2 排序。"""
    records: List[Dict[str, object]] = []
    checked = 0
    optional = [var for var in candidates if var not in base_vars]
    max_optional = max(0, max_vars - len(base_vars))

    for size in range(0, max_optional + 1):
        for combo in itertools.combinations(optional, size):
            features = list(base_vars) + list(combo)
            metrics = evaluate_model(df, features, standardize=standardize)
            if not metrics:
                continue
            records.append(metrics)
            checked += 1
            if max_combos and checked >= max_combos:
                break
        if max_combos and checked >= max_combos:
            break

    records.sort(key=lambda item: item.get("adj_r2", float("-inf")), reverse=True)
    return records


# === 辅助输出函数 ====================================================================

def render_features(columns: Sequence[str], reverse_map: Dict[str, str]) -> str:
    """将清洗后的列名转换回原始列名，并拼接字符串。"""
    original_names = [reverse_map.get(col, col) for col in columns]
    return ", ".join(original_names)


def enrich_results(
    results: Sequence[Dict[str, object]], reverse_map: Dict[str, str]
) -> pd.DataFrame:
    """将模型评估结果转换为 DataFrame，并添加原始列名列表。"""
    formatted = []
    for item in results:
        entry = item.copy()
        features = entry.pop("features", [])
        selected = entry.pop("selected_features", features)
        entry["features_clean"] = ", ".join(selected)
        entry["features_original"] = render_features(selected, reverse_map)
        formatted.append(entry)
    if not formatted:
        return pd.DataFrame()
    df = pd.DataFrame(formatted)
    df = df.sort_values(by="adj_r2", ascending=False)
    return df.reset_index(drop=True)


def print_summary(df: pd.DataFrame, limit: int = 5) -> None:
    """打印前几条结果简表。"""
    if df.empty:
        print("未找到满足条件的模型组合。")
        return
    display_cols = [
        "adj_r2",
        "r2",
        "rmse",
        "n_obs",
        "coef_x",
        "pvalue_x",
        "features_original",
    ]
    printable = df[display_cols].head(limit)
    with pd.option_context("display.max_rows", limit, "display.max_colwidth", 120):
        print(printable)


# === 主函数 ==========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="变量组合搜索与拟合度评估工具")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"数据文件路径（默认：{DEFAULT_DATA_PATH}）",
    )
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Excel 工作表名")
    parser.add_argument(
        "--method",
        type=str,
        choices=("stepwise", "subsets"),
        default="stepwise",
        help="变量组合搜索方法",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help="逗号分隔的候选变量分组（all 表示全部），例如 macro,innovation",
    )
    parser.add_argument(
        "--base-vars",
        type=str,
        default="",
        help="逗号分隔的基础控制变量（原始列名），始终包含在模型中",
    )
    parser.add_argument(
        "--extra-vars",
        type=str,
        default="",
        help="逗号分隔的额外候选变量（原始列名），可与分组同时使用",
    )
    parser.add_argument("--max-vars", type=int, default=6, help="模型中控制变量的最大数量")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.002,
        help="前向逐步回归中调整后 R^2 的最小增幅",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=200,
        help="子集枚举最多评估的组合数量（0 表示不限制）",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=0.4,
        help="控制变量允许的最大缺失率（0-1），超出将被剔除",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="在拟合前对候选控制变量进行标准化",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="保存结果的 CSV 文件路径",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """执行主流程并返回结果 DataFrame，便于在 notebook/测试中复用。"""
    args = parse_args(argv)

    df_raw, mapping = load_dataset(args.data, args.sheet)
    reverse_map = {v: k for k, v in mapping.items()}

    panel = prepare_panel(df_raw, mapping)

    # 处理候选分组
    if args.groups.strip().lower() == "all":
        group_keys = list(CANDIDATE_GROUPS.keys())
    else:
        group_keys = [key.strip() for key in args.groups.split(",") if key.strip()]
    unknown_groups = [g for g in group_keys if g not in CANDIDATE_GROUPS]
    if unknown_groups:
        raise KeyError(f"未定义的变量分组：{unknown_groups}")

    candidate_names: List[str] = []
    missing_by_group: Dict[str, List[str]] = {}

    for group in group_keys:
        names = CANDIDATE_GROUPS[group]
        resolved, missing = resolve_columns(names, mapping)
        candidate_names.extend(resolved)
        if missing:
            missing_by_group[group] = missing

    if args.extra_vars:
        extra_list = [item.strip() for item in args.extra_vars.split(",") if item.strip()]
        resolved_extra, missing_extra = resolve_columns(extra_list, mapping)
        candidate_names.extend(resolved_extra)
        if missing_extra:
            missing_by_group["extra"] = missing_extra

    # 基础变量处理
    base_vars: List[str] = []
    if args.base_vars:
        base_list = [item.strip() for item in args.base_vars.split(",") if item.strip()]
        resolved_base, missing_base = resolve_columns(base_list, mapping)
        base_vars.extend(resolved_base)
        if missing_base:
            missing_by_group["base"] = missing_base

    candidate_names = sorted(set(candidate_names) - {"y", "x", "country", "year"})
    base_vars = sorted(set(base_vars))

    if missing_by_group:
        print("以下原始列名未找到，请确认是否存在或需手动清洗：")
        for group, cols in missing_by_group.items():
            print(f"  - {group}: {', '.join(cols)}")

    # 按缺失率过滤
    if args.max_missing < 1.0:
        filtered_candidates = []
        for var in candidate_names:
            missing_ratio = 1.0 - float(panel[var].notna().mean())
            if missing_ratio <= args.max_missing:
                filtered_candidates.append(var)
        removed = sorted(set(candidate_names) - set(filtered_candidates))
        if removed:
            print("因缺失率较高而被剔除的变量：")
            for var in removed:
                print(f"  - {reverse_map.get(var, var)} (缺失率 {1.0 - panel[var].notna().mean():.2%})")
        candidate_names = filtered_candidates

    if not candidate_names and not base_vars:
        raise RuntimeError("候选变量集合为空，请检查参数。")

    if args.method == "stepwise":
        results = forward_stepwise(
            panel,
            candidate_names,
            base_vars,
            max_vars=args.max_vars,
            min_improvement=args.min_improvement,
            standardize=args.standardize,
        )
    else:
        results = enumerate_subsets(
            panel,
            candidate_names,
            base_vars,
            max_vars=args.max_vars,
            max_combos=None if args.max_combos <= 0 else args.max_combos,
            standardize=args.standardize,
        )

    df_results = enrich_results(results, reverse_map)
    print_summary(df_results)

    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"结果已保存到 {args.output}")

    return df_results


def main() -> None:
    run()


if __name__ == "__main__":
    main()

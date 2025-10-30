#!/usr/bin/env python3
"""
命令行工具：围绕机器人渗透率与全球价值链地位的变量筛选、面板回归与稳健性检验。

核心能力对应研究步骤：
1. 数据驱动变量筛选：缺失率与相关性过滤、VIF 共线性检测、LASSO/逐步筛选基准控制变量。
2. 模型估计：双重固定效应、滞后处理、聚类稳健标准误、机制/调节效应。
3. 门槛检验与稳健性分析：Hansen 风格门槛搜索、Bootstrap p 值、样本与口径替换。
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少依赖 statsmodels，无法执行共线性检测。请运行 `pip install statsmodels` 后重试。"
    ) from exc

# 默认数据路径指向项目 data 目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data.xlsx"

# 变量分组：可按需增删
CANDIDATE_GROUPS: Dict[str, List[str]] = {
    "macro": [
        "GDP（现价美元）",
        "人均 GDP（现价美元）",
        "人均 GDP增长（年增长率）",
        "固定资本形成总额（占 GDP 的百分比）",
        "资本形成总额（占 GDP 的百分比）",
        "国民总收入（GNI）（现价美元）",
    ],
    "industry": [
        "制造业，增加值（占GDP的百分比）",
        "制造业出口（占商品出口的百分比）",
        "工业增加值（占 GDP 的百分比）",
        "矿石和金属出口（占商品出口的百分比）",
        "高科技出口（占制成品出口的百分比）",
    ],
    "innovation": [
        "研发支出（占GDP的比例）",
        "R&D研究人员 （每百万人）",
        "每100万人中研发技术人员的数量",
        "科技期刊文章",
        "RD/劳动力",
        "RD/canyu",
        "RD/renkou",
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
        "利差（贷款利率减去存款利率，百分比）",
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
        "劳动力参与率，总数（占 15-64 岁总人口的百分比）（模拟劳工组织估计）",
    ],
}


@dataclass
class SelectionReport:
    kept: List[str]
    dropped_missing: Dict[str, float]
    dropped_corr: Dict[str, float]
    dropped_vif: Dict[str, float]
    method: str
    history: List[Dict[str, object]]


def sanitize_columns(columns: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    """将列名映射为安全的 snake_case，保证唯一性。"""
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
    """读取 Excel 并返回清洗后的 DataFrame 与映射。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")
    df_raw = pd.read_excel(path, sheet_name=sheet)
    sanitized_cols, mapping = sanitize_columns(df_raw.columns)
    df = df_raw.copy()
    df.columns = sanitized_cols
    return df, mapping


def prepare_panel(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """统一核心列名并清洗面板索引。"""
    rename_map = {
        mapping.get("y", "y"): "y",
        mapping.get("x", "x"): "x",
        mapping.get("country", "country"): "country",
        mapping.get("year", "year"): "year",
    }
    df = df.rename(columns=rename_map)
    missing_core = [col for col in ["y", "x", "country", "year"] if col not in df.columns]
    if missing_core:
        raise KeyError(f"缺少核心列：{missing_core}")
    df = df.dropna(subset=["y", "x", "country", "year"])
    df["year"] = df["year"].astype(int)
    df["country"] = df["country"].astype(str)
    return df


def resolve_columns(names: Sequence[str], mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    missing: List[str] = []
    for name in names:
        if name in mapping:
            resolved.append(mapping[name])
        else:
            missing.append(name)
    return resolved, missing


def filter_candidates_by_missing_and_corr(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    *,
    missing_threshold: float,
    corr_alpha: float,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """按缺失率和与 x/y 的相关性显著性过滤变量。"""
    kept: List[str] = []
    dropped_missing: Dict[str, float] = {}
    dropped_corr: Dict[str, float] = {}

    for col in candidate_cols:
        missing_ratio = 1.0 - float(df[col].notna().mean())
        if missing_ratio > missing_threshold:
            dropped_missing[col] = missing_ratio
            continue

        data = df[["y", "x", col]].dropna()
        if len(data) < 30:
            dropped_corr[col] = np.nan
            continue

        pvals = []
        for target in ("y", "x"):
            try:
                corr, pvalue = pearsonr(data[target], data[col])
            except Exception:
                corr, pvalue = np.nan, np.nan
            pvals.append(pvalue)
        if all((p is np.nan or p >= corr_alpha) for p in pvals):
            dropped_corr[col] = float(np.nanmean(pvals))
            continue
        kept.append(col)

    return kept, dropped_missing, dropped_corr


def compute_vif(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """计算指定列的 VIF。"""
    clean = df[list(columns)].dropna()
    if clean.empty or clean.shape[1] < 2:
        return pd.Series(dtype=float)
    X = clean.values
    vif_values = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X, i)
        vif_values.append(vif)
    return pd.Series(vif_values, index=columns)


def reduce_vif(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: float,
) -> Tuple[List[str], Dict[str, float]]:
    """迭代剔除 VIF 超阈值的变量。"""
    remaining = list(columns)
    removed: Dict[str, float] = {}
    while True:
        vif_series = compute_vif(df, remaining)
        if vif_series.empty:
            break
        max_vif = vif_series.max()
        if not math.isfinite(max_vif) or max_vif > threshold:
            target = vif_series.idxmax()
            removed[target] = float(max_vif if math.isfinite(max_vif) else np.nan)
            remaining.remove(target)
        else:
            break
        if len(remaining) <= 1:
            break
    return remaining, removed


def lasso_select_controls(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    max_controls: int,
    random_state: int = 42,
) -> Tuple[List[str], Dict[str, float]]:
    """使用 LASSO 选择控制变量。"""
    data = df[list(features) + ["y"]].dropna()
    if len(data) < len(features) + 5:
        return list(features[:max_controls]), {}

    X = data[features].values
    y = data["y"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LassoCV(cv=min(5, len(data) // 5) or 3, random_state=random_state)
        model.fit(X_scaled, y)

    coefs = pd.Series(np.abs(model.coef_), index=features)
    selected = coefs[coefs > 0].sort_values(ascending=False).index.tolist()
    if not selected:
        selected = coefs.sort_values(ascending=False).index.tolist()
    return selected[:max_controls], coefs.to_dict()


def forward_stepwise(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    base_vars: Sequence[str],
    max_vars: int,
    standardize: bool = False,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """基于调整后 R² 的前向逐步回归。"""
    selected = list(base_vars)
    remaining = [f for f in features if f not in selected]
    history: List[Dict[str, object]] = []
    current_score: Optional[float] = None

    while remaining and len(selected) < max_vars:
        best_candidate: Optional[str] = None
        best_metrics: Optional[Dict[str, object]] = None
        for candidate in remaining:
            trial = selected + [candidate]
            metrics = evaluate_model(df, trial, standardize=standardize)
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
        if current_score is not None and improvement <= 0:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_score = best_metrics["adj_r2"]
        history.append(best_metrics | {"selected": list(selected)})

    return selected, history


def evaluate_model(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    standardize: bool = False,
    min_obs: int = 60,
) -> Optional[Dict[str, object]]:
    """用于逐步回归的评估函数，保持双重固定效应设定。"""
    subset_cols = ["y", "x", "country", "year"] + list(features)
    data = df[subset_cols].dropna()
    if len(data) < min_obs or data["country"].nunique() < 2 or data["year"].nunique() < 2:
        return None
    if standardize and features:
        scaler = StandardScaler()
        data = data.copy()
        data[features] = scaler.fit_transform(data[features])

    formula = "y ~ x"
    if features:
        formula += " + " + " + ".join(features)
    formula += " + C(country) + C(year)"

    global _SMF
    if "_SMF" not in globals() or globals()["_SMF"] is None:
        try:
            import statsmodels.formula.api as smf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "缺少依赖 statsmodels，请运行 `pip install statsmodels` 后再执行脚本。"
            ) from exc
        globals()["_SMF"] = smf
    else:
        smf = globals()["_SMF"]  # type: ignore[assignment]

    model = smf.ols(formula=formula, data=data)
    result = model.fit()
    residuals = result.resid
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    metrics: Dict[str, object] = {
        "n_obs": int(len(data)),
        "r2": float(result.rsquared),
        "adj_r2": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "rmse": rmse,
        "features": list(features),
    }
    return metrics


def select_controls_pipeline(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    *,
    missing_threshold: float,
    corr_alpha: float,
    vif_threshold: float,
    selection_method: str,
    max_controls: int,
    base_vars: Sequence[str],
) -> SelectionReport:
    """按照既定规则筛选控制变量。"""
    filtered, dropped_missing, dropped_corr = filter_candidates_by_missing_and_corr(
        df, candidate_cols, missing_threshold=missing_threshold, corr_alpha=corr_alpha
    )

    after_vif, dropped_vif = reduce_vif(df, filtered, threshold=vif_threshold)

    if selection_method == "lasso":
        selected, diagnostics = lasso_select_controls(
            df, after_vif, max_controls=max_controls - len(base_vars)
        )
        selected = base_vars + [col for col in selected if col not in base_vars]
        history = [{"method": "lasso", "coeff_abs": diagnostics}]
    else:
        selected, step_history = forward_stepwise(
            df,
            after_vif,
            base_vars=base_vars,
            max_vars=max_controls,
        )
        history = [{"method": "stepwise", "steps": step_history}]

    return SelectionReport(
        kept=selected[:max_controls],
        dropped_missing=dropped_missing,
        dropped_corr=dropped_corr,
        dropped_vif=dropped_vif,
        method=selection_method,
        history=history,
    )


def create_lagged_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    lag: int,
    group_col: str = "country",
    time_col: str = "year",
) -> pd.DataFrame:
    """为指定列创建滞后项。"""
    df = df.sort_values([group_col, time_col]).copy()
    for col in columns:
        lagged = df.groupby(group_col)[col].shift(lag)
        df[f"{col}_lag{lag}"] = lagged
    return df


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = series.quantile(lower), series.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)


def apply_transformations(
    df: pd.DataFrame,
    *,
    log_cols: Sequence[str],
    winsorize: bool = True,
    winsor_limits: Tuple[float, float] = (0.01, 0.99),
    add_log1p_for_x: bool = True,
) -> pd.DataFrame:
    """执行金额对数、缩尾以及 x 的 ln(1+x) 变换。"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    lower, upper = winsor_limits

    if winsorize:
        for col in numeric_cols:
            df[col] = winsorize_series(df[col], lower=lower, upper=upper)

    for col in log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log(df[col].clip(lower=1e-9))

    if add_log1p_for_x and "x" in df.columns:
        df["x_log1p"] = np.log1p(df["x"].clip(lower=0))

    return df


def fit_fixed_effects_model(
    df: pd.DataFrame,
    *,
    dependent: str,
    regressor: str,
    controls: Sequence[str],
    cluster_col: str,
) -> Tuple[object, pd.DataFrame]:
    """拟合双重固定效应并返回聚类稳健结果。"""
    formula = f"{dependent} ~ {regressor}"
    if controls:
        formula += " + " + " + ".join(controls)
    formula += " + C(country) + C(year)"

    if "_SMF" not in globals() or globals()["_SMF"] is None:
        try:
            import statsmodels.formula.api as smf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "缺少依赖 statsmodels，请运行 `pip install statsmodels` 后再执行脚本。"
            ) from exc
        globals()["_SMF"] = smf
    else:
        smf = globals()["_SMF"]  # type: ignore[assignment]

    model = smf.ols(formula=formula, data=df)
    fitted = model.fit()
    robust = fitted.get_robustcov_results(cov_type="cluster", groups=df[cluster_col])

    summary_df = pd.DataFrame(
        {
            "coef": robust.params,
            "std_err": robust.bse,
            "t": robust.tvalues,
            "p>|t|": robust.pvalues,
        }
    )
    return robust, summary_df


def run_baseline_estimation(
    df: pd.DataFrame,
    *,
    controls: Sequence[str],
    lag_x: int,
    cluster_col: str = "country",
) -> Dict[str, object]:
    """基准模型：x 滞后 + 双重固定效应 + 聚类稳健标准误。"""
    work = create_lagged_features(df, ["x"], lag=lag_x)
    regressor = f"x_lag{lag_x}"
    work = work.dropna(subset=[regressor])

    model, table = fit_fixed_effects_model(
        work,
        dependent="y",
        regressor=regressor,
        controls=controls,
        cluster_col=cluster_col,
    )
    return {
        "model": model,
        "table": table,
        "regressor": regressor,
        "n_obs": int(model.nobs),
    }


def run_mediation_analysis(
    df: pd.DataFrame,
    *,
    mediator: str,
    controls: Sequence[str],
    lag_x: int,
) -> Dict[str, object]:
    """两步中介效应：x->中介、y->x+中介。"""
    work = create_lagged_features(df, ["x"], lag=lag_x).dropna(subset=[f"x_lag{lag_x}"])
    mediator_formula_controls = [col for col in controls if col != mediator]

    med_result, med_table = fit_fixed_effects_model(
        work,
        dependent=mediator,
        regressor=f"x_lag{lag_x}",
        controls=mediator_formula_controls,
        cluster_col="country",
    )

    outcome_controls = [mediator] + mediator_formula_controls
    out_result, out_table = fit_fixed_effects_model(
        work,
        dependent="y",
        regressor=f"x_lag{lag_x}",
        controls=outcome_controls,
        cluster_col="country",
    )

    return {
        "mediator_model": med_result,
        "mediator_table": med_table,
        "outcome_model": out_result,
        "outcome_table": out_table,
    }


def run_interaction_analysis(
    df: pd.DataFrame,
    *,
    moderator: str,
    controls: Sequence[str],
    lag_x: int,
) -> Dict[str, object]:
    """交互调节：加入 x_lag * moderator。"""
    work = create_lagged_features(df, ["x"], lag=lag_x).dropna(subset=[f"x_lag{lag_x}", moderator])
    work[f"x_lag{lag_x}_x_{moderator}"] = work[f"x_lag{lag_x}"] * work[moderator]
    extra_controls = [col for col in controls if col != moderator]

    model, table = fit_fixed_effects_model(
        work,
        dependent="y",
        regressor=f"x_lag{lag_x}",
        controls=[moderator, f"x_lag{lag_x}_x_{moderator}"] + extra_controls,
        cluster_col="country",
    )
    return {"interaction_model": model, "interaction_table": table}


def panel_threshold_analysis(
    df: pd.DataFrame,
    *,
    threshold_var: str,
    controls: Sequence[str],
    lag_x: int,
    grid_points: int = 10,
    bootstrap: int = 100,
) -> Dict[str, object]:
    """Hansen 面板门槛的简化实现：搜索阈值并自举 p 值。"""
    work = create_lagged_features(df, ["x"], lag=lag_x).dropna(subset=[f"x_lag{lag_x}", threshold_var])
    baseline_model, _ = fit_fixed_effects_model(
        work,
        dependent="y",
        regressor=f"x_lag{lag_x}",
        controls=controls,
        cluster_col="country",
    )
    baseline_ssr = float(np.square(baseline_model.resid).sum())

    thresholds = (
        work[threshold_var]
        .quantile(np.linspace(0.2, 0.8, grid_points))
        .drop_duplicates()
        .tolist()
    )

    search_records = []
    best_model = None
    best_resid = None
    best_gamma = None

    for gamma in thresholds:
        data = work.copy()
        data["regime_low"] = (data[threshold_var] <= gamma).astype(int)
        data["regime_high"] = 1 - data["regime_low"]
        data[f"x_lag{lag_x}_low"] = data[f"x_lag{lag_x}"] * data["regime_low"]
        data[f"x_lag{lag_x}_high"] = data[f"x_lag{lag_x}"] * data["regime_high"]

        model, _ = fit_fixed_effects_model(
            data,
            dependent="y",
            regressor=f"x_lag{lag_x}_low",
            controls=[f"x_lag{lag_x}_high"] + list(controls),
            cluster_col="country",
        )
        ssr = float(np.square(model.resid).sum())
        search_records.append({"gamma": gamma, "ssr": ssr})
        if best_resid is None or ssr < best_resid:
            best_resid = ssr
            best_gamma = gamma
            best_model = model

    lr_stat = max(baseline_ssr - (best_resid or baseline_ssr), 0)

    # Bootstrap p 值：按国家重抽样
    rng = np.random.default_rng(42)
    countries = work["country"].unique()
    boot_stats = []
    for _ in range(bootstrap):
        sampled = rng.choice(countries, size=len(countries), replace=True)
        sampled_df = pd.concat([work[work["country"] == c] for c in sampled], ignore_index=True)
        if sampled_df.empty:
            continue
        baseline_b, _ = fit_fixed_effects_model(
            sampled_df,
            dependent="y",
            regressor=f"x_lag{lag_x}",
            controls=controls,
            cluster_col="country",
        )
        baseline_ssr_b = float(np.square(baseline_b.resid).sum())
        best_ssr_b = None
        for gamma in thresholds:
            data_b = sampled_df.copy()
            data_b["regime_low"] = (data_b[threshold_var] <= gamma).astype(int)
            data_b["regime_high"] = 1 - data_b["regime_low"]
            data_b[f"x_lag{lag_x}_low"] = data_b[f"x_lag{lag_x}"] * data_b["regime_low"]
            data_b[f"x_lag{lag_x}_high"] = data_b[f"x_lag{lag_x}"] * data_b["regime_high"]
            model_b, _ = fit_fixed_effects_model(
                data_b,
                dependent="y",
                regressor=f"x_lag{lag_x}_low",
                controls=[f"x_lag{lag_x}_high"] + list(controls),
                cluster_col="country",
            )
            ssr_b = float(np.square(model_b.resid).sum())
            if best_ssr_b is None or ssr_b < best_ssr_b:
                best_ssr_b = ssr_b
        if best_ssr_b is not None:
            boot_stats.append(max(baseline_ssr_b - best_ssr_b, 0))

    p_value = (
        float(np.mean([stat >= lr_stat for stat in boot_stats])) if boot_stats else np.nan
    )

    return {
        "best_threshold": best_gamma,
        "lr_stat": lr_stat,
        "bootstrap_p": p_value,
        "search_records": pd.DataFrame(search_records),
        "model": best_model,
    }


def linear_interpolate_controls(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """按国家-年份对控制变量线性插值。"""
    df = df.sort_values(["country", "year"]).copy()
    for col in columns:
        df[col] = df.groupby("country")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    return df


def mice_impute_controls(df: pd.DataFrame, columns: Sequence[str], random_state: int = 42) -> pd.DataFrame:
    """使用多重链式方程（MICE）对控制变量插补。"""
    imputer = IterativeImputer(random_state=random_state, max_iter=10, sample_posterior=False)
    work = df.copy()
    mask = work[columns].notna().all(axis=1)
    filled = imputer.fit_transform(work[columns])
    work[columns] = filled
    return work


def run_robustness_checks(
    df: pd.DataFrame,
    *,
    controls: Sequence[str],
    lag_options: Sequence[int],
    alt_x_columns: Sequence[str],
    impute: bool = True,
) -> pd.DataFrame:
    """执行多种稳健性设定并汇总估计结果。"""
    records: List[Dict[str, object]] = []
    base_df = df.copy()

    if impute:
        linear_df = linear_interpolate_controls(base_df, controls)
        mice_df = mice_impute_controls(base_df, controls)
        datasets = [
            ("original", base_df),
            ("linear_interp", linear_df),
            ("mice", mice_df),
        ]
    else:
        datasets = [("original", base_df)]

    for label, data in datasets:
        for lag in lag_options:
            work = create_lagged_features(data, ["x"], lag=lag).dropna(subset=[f"x_lag{lag}"])
            model, table = fit_fixed_effects_model(
                work,
                dependent="y",
                regressor=f"x_lag{lag}",
                controls=controls,
                cluster_col="country",
            )
            records.append(
                {
                    "variant": f"{label}_lag{lag}",
                    "coef": float(model.params.get(f"x_lag{lag}", np.nan)),
                    "se": float(model.bse.get(f"x_lag{lag}", np.nan)),
                    "pvalue": float(model.pvalues.get(f"x_lag{lag}", np.nan)),
                    "n_obs": int(model.nobs),
                }
            )

        for alt_x in alt_x_columns:
            if alt_x not in data.columns:
                continue
            model, table = fit_fixed_effects_model(
                data.dropna(subset=[alt_x]),
                dependent="y",
                regressor=alt_x,
                controls=controls,
                cluster_col="country",
            )
            records.append(
                {
                    "variant": f"{label}_{alt_x}",
                    "coef": float(model.params.get(alt_x, np.nan)),
                    "se": float(model.bse.get(alt_x, np.nan)),
                    "pvalue": float(model.pvalues.get(alt_x, np.nan)),
                    "n_obs": int(model.nobs),
                }
            )

    return pd.DataFrame(records)


def print_table(df: pd.DataFrame, head: int = 10) -> None:
    if df.empty:
        print("（无数据）")
        return
    with pd.option_context("display.max_rows", head, "display.max_columns", None, "display.width", 120):
        print(df.head(head))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="机器人渗透率与价值链分析工具")
    parser.add_argument("--task", choices=["select", "estimate", "mechanism", "threshold", "robustness"], default="select")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help=f"数据文件路径（默认：{DEFAULT_DATA_PATH}）")
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Excel 工作表名")
    parser.add_argument("--groups", type=str, default="all", help="候选变量分组，逗号分隔，all 表示全部")
    parser.add_argument("--extra-vars", type=str, default="", help="额外候选变量，逗号分隔")
    parser.add_argument("--base-vars", type=str, default="", help="基准控制变量，逗号分隔")
    parser.add_argument("--missing-threshold", type=float, default=0.3, help="缺失率阈值")
    parser.add_argument("--corr-alpha", type=float, default=0.05, help="相关性显著性水平")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF 剔除阈值")
    parser.add_argument("--selection-method", choices=["lasso", "stepwise"], default="lasso")
    parser.add_argument("--max-controls", type=int, default=8, help="最多控制变量数量")
    parser.add_argument("--lag-x", type=int, default=1, help="x 的滞后期数")
    parser.add_argument("--mediator", type=str, default="", help="中介变量名称（原始列名）")
    parser.add_argument("--moderator", type=str, default="", help="调节变量名称（原始列名）")
    parser.add_argument("--threshold-var", type=str, default="", help="门槛变量名称（原始列名）")
    parser.add_argument("--threshold-bootstrap", type=int, default=100, help="门槛检验 Bootstrap 次数")
    parser.add_argument("--output", type=Path, default=None, help="结果输出 CSV/JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="打印额外诊断信息")
    return parser.parse_args(argv)


def build_candidate_list(groups: Sequence[str], mapping: Dict[str, str], extra_vars: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    missing: Dict[str, List[str]] = {}
    candidates: List[str] = []
    for group in groups:
        names = CANDIDATE_GROUPS.get(group, [])
        resolved, missing_cols = resolve_columns(names, mapping)
        candidates.extend(resolved)
        if missing_cols:
            missing[group] = missing_cols
    if extra_vars:
        resolved_extra, missing_extra = resolve_columns(extra_vars, mapping)
        candidates.extend(resolved_extra)
        if missing_extra:
            missing["extra"] = missing_extra
    return sorted(set(candidates) - {"y", "x", "country", "year"}), missing


def run(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)

    df_raw, mapping = load_dataset(args.data, args.sheet)
    panel = prepare_panel(df_raw, mapping)

    # 添加常用变换：x -> ln(1+x)，并可扩展
    panel = apply_transformations(panel, log_cols=[], winsorize=True, add_log1p_for_x=True)

    # 构建候选变量列表
    if args.groups.lower() == "all":
        group_keys = list(CANDIDATE_GROUPS.keys())
    else:
        group_keys = [g.strip() for g in args.groups.split(",") if g.strip()]
    extra_vars = [v.strip() for v in args.extra_vars.split(",") if v.strip()]
    base_vars_raw = [v.strip() for v in args.base_vars.split(",") if v.strip()]

    candidates, missing_info = build_candidate_list(group_keys, mapping, extra_vars)
    base_vars, base_missing = resolve_columns(base_vars_raw, mapping)
    if base_missing:
        missing_info["base"] = base_missing

    if missing_info:
        print("以下原始列未在数据中找到，将跳过：")
        for key, cols in missing_info.items():
            print(f"  - {key}: {', '.join(cols)}")

    if args.task == "select":
        report = select_controls_pipeline(
            panel,
            candidates,
            missing_threshold=args.missing_threshold,
            corr_alpha=args.corr_alpha,
            vif_threshold=args.vif_threshold,
            selection_method=args.selection_method,
            max_controls=args.max_controls,
            base_vars=base_vars,
        )
        if args.verbose:
            print("缺失率剔除：", report.dropped_missing)
            print("相关性剔除：", report.dropped_corr)
            print("VIF 剔除：", report.dropped_vif)
            print("筛选历史：", report.history)
        print("推荐控制变量：", ", ".join(report.kept))
        result_df = pd.DataFrame({"selected_controls": report.kept})

    elif args.task == "estimate":
        selected_controls = base_vars or candidates[: args.max_controls]
        baseline = run_baseline_estimation(panel, controls=selected_controls, lag_x=args.lag_x)
        print(f"基准模型（滞后 {args.lag_x}）共 {baseline['n_obs']} 条样本，关键系数：")
        print_table(baseline["table"].loc[[baseline["regressor"]]])
        result_df = baseline["table"]

    elif args.task == "mechanism":
        if not args.mediator and not args.moderator:
            raise SystemExit("机制分析需指定 --mediator 或 --moderator")
        selected_controls = base_vars or candidates[: args.max_controls]
        outputs: Dict[str, pd.DataFrame] = {}
        if args.mediator:
            mediator_col, missing_med = resolve_columns([args.mediator], mapping)
            if missing_med:
                raise SystemExit(f"未找到中介变量：{args.mediator}")
            mediator_report = run_mediation_analysis(
                panel,
                mediator=mediator_col[0],
                controls=selected_controls,
                lag_x=args.lag_x,
            )
            print("中介模型：")
            print_table(mediator_report["mediator_table"].loc[[f"x_lag{args.lag_x}"]])
            print("结果方程：")
            print_table(
                mediator_report["outcome_table"].loc[
                    [f"x_lag{args.lag_x}", mediator_col[0]]
                ]
            )
            outputs["mediator"] = mediator_report["outcome_table"]
        if args.moderator:
            moderator_col, missing_mod = resolve_columns([args.moderator], mapping)
            if missing_mod:
                raise SystemExit(f"未找到调节变量：{args.moderator}")
            interaction_report = run_interaction_analysis(
                panel,
                moderator=moderator_col[0],
                controls=selected_controls,
                lag_x=args.lag_x,
            )
            print("交互模型：")
            print_table(
                interaction_report["interaction_table"].loc[
                    [f"x_lag{args.lag_x}", moderator_col[0], f"x_lag{args.lag_x}_x_{moderator_col[0]}"]
                ]
            )
            outputs["interaction"] = interaction_report["interaction_table"]
        result_df = pd.concat(outputs.values(), keys=outputs.keys()) if outputs else pd.DataFrame()

    elif args.task == "threshold":
        if not args.threshold_var:
            raise SystemExit("门槛分析需指定 --threshold-var")
        threshold_col, missing_thr = resolve_columns([args.threshold_var], mapping)
        if missing_thr:
            raise SystemExit(f"未找到门槛变量：{args.threshold_var}")
        selected_controls = base_vars or candidates[: args.max_controls]
        report = panel_threshold_analysis(
            panel,
            threshold_var=threshold_col[0],
            controls=selected_controls,
            lag_x=args.lag_x,
            bootstrap=args.threshold_bootstrap,
        )
        print(f"最佳门槛值：{report['best_threshold']:.4f}")
        print(f"LR 统计量：{report['lr_stat']:.4f}")
        print(f"Bootstrap p-value：{report['bootstrap_p']:.4f}")
        result_df = report["search_records"]

    else:  # robustness
        selected_controls = base_vars or candidates[: args.max_controls]
        alt_x = ["x_log1p"]
        report = run_robustness_checks(
            panel,
            controls=selected_controls,
            lag_options=[args.lag_x, args.lag_x + 1],
            alt_x_columns=alt_x,
            impute=True,
        )
        print("稳健性检验结果：")
        print_table(report, head=len(report))
        result_df = report

    if args.output:
        if args.output.suffix.lower() == ".csv":
            result_df.to_csv(args.output, index=False)
        elif args.output.suffix.lower() in {".json", ".jsonl"}:
            result_df.to_json(args.output, orient="records", force_ascii=False, indent=2)
        else:
            raise SystemExit("仅支持 CSV 或 JSON 输出")
        print(f"结果已保存至 {args.output}")

    return result_df


def main() -> None:
    run()


if __name__ == "__main__":
    main()

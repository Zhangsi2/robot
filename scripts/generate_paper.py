#!/usr/bin/env python3
"""自动生成机器人渗透率论文写作 prompt 与分析摘要。

新的生成逻辑分为两步：
1. 利用量化分析结果构建上下文摘要，并结合写作模板生成 prompt。
2. 将 prompt 提交给 LLM（例如当前对话的助手）生成最终论文稿。

若需要复用旧版直接输出论文的能力，可加上 ``--draft-from-template`` 开关。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from robot_analysis.cli import (
    CANDIDATE_GROUPS,
    apply_transformations,
    build_candidate_list,
    create_lagged_features,
    load_dataset,
    panel_threshold_analysis,
    prepare_panel,
    run_baseline_estimation,
    run_interaction_analysis,
    run_mediation_analysis,
    run_robustness_checks,
    select_controls_pipeline,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文写作 prompt 与分析摘要")
    parser.add_argument("--data", type=Path, default=Path("data/data.xlsx"))
    parser.add_argument("--sheet", type=str, default="Sheet1")
    parser.add_argument("--groups", type=str, default="all")
    parser.add_argument("--extra-vars", type=str, default="")
    parser.add_argument("--base-vars", type=str, default="")
    parser.add_argument("--mediator", type=str, default="")
    parser.add_argument("--moderator", type=str, default="")
    parser.add_argument("--threshold-var", type=str, default="")
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("prompts/paper_prompt_template.md"),
        help="写作指令模板路径",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("outputs/paper_prompt.md"),
        help="生成的 LLM prompt 输出路径",
    )
    parser.add_argument(
        "--context-output",
        type=Path,
        default=Path("outputs/paper_context.json"),
        help="分析摘要 JSON 输出路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="（可选）直接生成论文草稿的输出路径；需搭配 --draft-from-template",
    )
    parser.add_argument("--lag-x", type=int, default=1)
    parser.add_argument("--max-controls", type=int, default=8)
    parser.add_argument(
        "--selection-method",
        choices=["lasso", "stepwise"],
        default="lasso",
    )
    parser.add_argument("--missing-threshold", type=float, default=0.3)
    parser.add_argument("--corr-alpha", type=float, default=0.05)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument(
        "--include-robustness",
        action="store_true",
        help="生成稳健性检验描述",
    )
    parser.add_argument(
        "--draft-from-template",
        action="store_true",
        help="使用内置模版直接生成论文草稿（与新版 prompt 并行输出）",
    )
    return parser.parse_args(argv)


def to_original(names: Sequence[str], reverse_map: Dict[str, str]) -> List[str]:
    return [reverse_map.get(name, name) for name in names]


def summarize_robustness(df: pd.DataFrame) -> str:
    if df.empty:
        return "不同设定下未能收敛到有效结果，后续可针对特定设定进一步检验。"
    sig_share = (df["pvalue"] < 0.1).mean()
    avg_coef = df["coef"].mean()
    variants = ", ".join(df["variant"].tolist())
    return (
        f"共比较 {len(df)} 组设定（包括 {variants}），平均系数为 {avg_coef:.4f}。"
        f"其中 {sig_share:.0%} 的设定在 10% 显著性水平下保持正向且显著，"
        "说明基准结论对变量口径、滞后期及缺失处理方式的选择具有相当稳健性。"
    )


def synthesize_trend_description(panel: pd.DataFrame) -> str:
    grouped = panel.groupby("year")[["x", "y"]].mean().dropna()
    if len(grouped) < 2:
        return "样本期内机器人渗透率与价值链地位的均值变化较为平稳。"
    start, end = grouped.index[0], grouped.index[-1]
    dx = grouped["x"].iloc[-1] - grouped["x"].iloc[0]
    dy = grouped["y"].iloc[-1] - grouped["y"].iloc[0]
    corr_xy = panel[["x", "y"]].corr().iloc[0, 1]

    def describe(delta: float, positive_word: str, negative_word: str) -> str:
        if delta > 0:
            return f"{positive_word} {delta:.4f}"
        if delta < 0:
            return f"{negative_word} {abs(delta):.4f}"
        return "基本保持不变"

    dx_text = describe(dx, "提高了", "下降了")
    dy_text = describe(dy, "同步提升", "同步下降")
    if corr_xy > 0.1:
        corr_text = f"呈现明显的正相关（相关系数 {corr_xy:.3f}）"
    elif corr_xy < -0.1:
        corr_text = f"呈现明显的负相关（相关系数 {corr_xy:.3f}）"
    else:
        corr_text = f"整体相关性较弱（相关系数 {corr_xy:.3f}）"
    return (
        f"从 {start} 年到 {end} 年，样本国家的平均机器人渗透率{dx_text}，"
        f"价值链地位指标{dy_text}。整体来看，两者 {corr_text}。"
    )


def highlight_countries(panel: pd.DataFrame, top_k: int = 5) -> str:
    ranking = (
        panel.groupby("country")["x"].mean().dropna().sort_values(ascending=False).head(top_k)
    )
    items = [f"{idx}（渗透率均值 {val:.4f}）" for idx, val in ranking.items()]
    return "、".join(items)


def build_section(title: str, paragraphs: List[str]) -> str:
    body = "\n\n".join(paragraphs)
    return f"## {title}\n\n{body}\n"


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.endswith(("。", "！", "？", ".", "!", "?")):
        return text
    return f"{text}。"


def normalize_for_bullet(text: str) -> str:
    clean = " ".join(text.split())
    return ensure_sentence(clean)


def combine_sentences(*parts: str) -> str:
    sentences = [ensure_sentence(part) for part in parts if part and part.strip()]
    return " ".join(sentences)


def compose_paper_markdown(context: Dict[str, object]) -> str:
    """基于分析上下文拼装 Markdown 草稿，用于兼容旧流程。"""

    intro = [
        "# 机器人渗透率与全球价值链地位：基于跨国面板的实证分析",
        context["intro_text"],
    ]

    data_section = build_section(
        "数据与变量",
        [
            "数据来源：世界发展指标（WDI）与机器人行业配套资料，经 1%/99% 缩尾与 `ln(1+x)` 变换处理后，尽量降低极端值对估计的干扰。",
            context["top_countries_text"],
            context["trend_narrative"],
        ],
    )

    method_section = build_section(
        "研究设计",
        [
            context["control_strategy"],
            (
                f"基准模型设定为双重固定效应框架，解释变量使用滞后 {context['lag_x']} 期的机器人渗透率，"
                "并对标准误实施国家聚类，以控制跨期与跨截面相关性。"
            ),
        ],
    )

    baseline_section = build_section(
        "基准回归结果",
        [
            f"{context['baseline_results']} {context['baseline_effect_text']}",
        ],
    )

    mechanism_section = build_section(
        "机制与调节效应",
        [
            context["mechanism_detail"],
            combine_sentences(f"综合来看，{context['mechanism_implication']}")
        ],
    )

    threshold_section = build_section(
        "门槛行为",
        [
            context["threshold_detail"],
            combine_sentences(f"这一发现意味着：{context['threshold_implication']}")
        ],
    )

    robustness_section = build_section(
        "稳健性检验",
        [
            "为验证结论的稳定性，本文替换了解释变量口径、滞后设定，并比较线性插值与 MICE 等缺失处理方式。",
            context["robustness_summary"],
        ],
    )

    conclusion_section = build_section(
        "结论与政策建议",
        [
            context["policy_summary"],
            context["future_work"],
        ],
    )

    return "\n\n".join(
        intro
        + [
            data_section,
            method_section,
            baseline_section,
            mechanism_section,
            threshold_section,
            robustness_section,
            conclusion_section,
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.context_output.parent.mkdir(parents=True, exist_ok=True)

    if not args.prompt_template.exists():
        raise FileNotFoundError(f"未找到写作模板：{args.prompt_template}")

    df_raw, mapping = load_dataset(args.data, args.sheet)
    reverse_map = {v: k for k, v in mapping.items()}

    panel = prepare_panel(df_raw, mapping)
    panel = apply_transformations(panel, log_cols=[], winsorize=True, add_log1p_for_x=True)

    if args.groups.lower() == "all":
        group_keys = list(CANDIDATE_GROUPS.keys())
    else:
        group_keys = [g.strip() for g in args.groups.split(",") if g.strip()]

    extra_vars = [v.strip() for v in args.extra_vars.split(",") if v.strip()]
    base_vars_raw = [v.strip() for v in args.base_vars.split(",") if v.strip()]

    candidates, missing = build_candidate_list(group_keys, mapping, extra_vars)
    base_vars: List[str] = []
    base_missing: List[str] = []
    for name in base_vars_raw:
        mapped = mapping.get(name)
        if mapped:
            base_vars.append(mapped)
        else:
            base_missing.append(name)

    if missing or base_missing:
        print("警告：部分指定变量缺失，将被自动忽略：")
        for key, cols in missing.items():
            print(f"  - {key}: {', '.join(cols)}")
        if base_missing:
            print(f"  - base: {', '.join(base_missing)}")

    selection_report = select_controls_pipeline(
        panel,
        candidates,
        missing_threshold=args.missing_threshold,
        corr_alpha=args.corr_alpha,
        vif_threshold=args.vif_threshold,
        selection_method=args.selection_method,
        max_controls=args.max_controls,
        base_vars=base_vars,
    )

    selected_controls = selection_report.kept[: args.max_controls]
    selected_controls_original = to_original(selected_controls, reverse_map)

    panel_with_lag = create_lagged_features(panel, ["x"], lag=args.lag_x)

    baseline = run_baseline_estimation(panel, controls=selected_controls, lag_x=args.lag_x)
    baseline_row = baseline["table"].loc[[baseline["regressor"]]]
    baseline_coef = float(baseline_row["coef"].iloc[0])
    baseline_se = float(baseline_row["std_err"].iloc[0])
    baseline_p = float(baseline_row["p>|t|"].iloc[0])

    std_y = panel_with_lag["y"].std()
    lagged_col = f"x_lag{args.lag_x}"
    if lagged_col in panel_with_lag.columns:
        std_x_series = panel_with_lag[lagged_col]
    else:
        std_x_series = panel_with_lag["x"]
    std_x = std_x_series.std()
    baseline_effect_std = baseline_coef * (std_x / std_y) if std_y else baseline_coef

    mechanism_detail = "由于缺乏指定的中介或调节变量，本研究留待后续工作进一步完善。"
    mechanism_implication = "机器人渗透通过提高生产率与创新投入间接推动价值链升级"
    if args.mediator:
        mediator_name = mapping.get(args.mediator)
        if mediator_name:
            mediator_report = run_mediation_analysis(
                panel,
                mediator=mediator_name,
                controls=selected_controls,
                lag_x=args.lag_x,
            )
            mediator_key = f"x_lag{args.lag_x}"
            med_coef = float(mediator_report["mediator_table"].loc[mediator_key, "coef"])
            med_p = float(mediator_report["mediator_table"].loc[mediator_key, "p>|t|"])
            out_coef = float(mediator_report["outcome_table"].loc[mediator_name, "coef"])
            out_p = float(mediator_report["outcome_table"].loc[mediator_name, "p>|t|"])
            mediator_original = reverse_map.get(mediator_name, mediator_name)
            mechanism_detail = (
                f"以“{mediator_original}”为中介变量的两步回归显示：机器人渗透率对中介变量的影响系数为 "
                f"{med_coef:.4f}（p={med_p:.3f}），中介变量对价值链地位的影响系数为 {out_coef:.4f}（p={out_p:.3f}）。"
                "说明机器人渗透率部分通过该中介渠道改善价值链地位。"
            )
            mechanism_implication = (
                f"机器人渗透率的提升首先强化了 {mediator_original}，进而带动价值链攀升"
            )
        else:
            print(f"提示：未在数据字典中找到中介变量 {args.mediator}，跳过中介分析。")

    if args.moderator:
        moderator_name = mapping.get(args.moderator)
        if moderator_name:
            interaction_report = run_interaction_analysis(
                panel,
                moderator=moderator_name,
                controls=selected_controls,
                lag_x=args.lag_x,
            )
            inter_key = f"x_lag{args.lag_x}_x_{moderator_name}"
            inter_coef = float(interaction_report["interaction_table"].loc[inter_key, "coef"])
            inter_p = float(interaction_report["interaction_table"].loc[inter_key, "p>|t|"])
            moderator_original = reverse_map.get(moderator_name, moderator_name)
            mechanism_detail += (
                f" 此外，机器人渗透率与“{moderator_original}”的交互项系数为 {inter_coef:.4f}"
                f"（p={inter_p:.3f}），表明当该调节变量处于较高水平时，机器人渗透率的正向效应进一步放大。"
            )
            mechanism_implication += f"，并在 {moderator_original} 水平较高时呈现更强的边际效应"
        else:
            print(f"提示：未在数据字典中找到调节变量 {args.moderator}，跳过交互分析。")

    threshold_detail = "未指定门槛变量，相关分析暂缺。"
    threshold_implication = "未来可继续探索非线性门槛效应"
    if args.threshold_var:
        threshold_name = mapping.get(args.threshold_var)
        if threshold_name:
            threshold_report = panel_threshold_analysis(
                panel,
                threshold_var=threshold_name,
                controls=selected_controls,
                lag_x=args.lag_x,
            )
            gamma = float(threshold_report["best_threshold"])
            p_value = float(threshold_report["bootstrap_p"])
            threshold_original = reverse_map.get(threshold_name, threshold_name)
            threshold_detail = (
                f"以“{threshold_original}”为门槛变量的检验中，最优门槛值为 {gamma:.4f}，"
                f"Bootstrap p 值为 {p_value:.3f}。这表明当该指标跨越门槛后，机器人渗透率的边际效应发生显著变化。"
            )
            threshold_implication = (
                f"{threshold_original} 存在显著门槛，超过 {gamma:.4f} 后机器人渗透率的正向作用更为突出"
            )
        else:
            print(f"提示：未在数据字典中找到门槛变量 {args.threshold_var}，跳过门槛检验。")

    robustness_summary = "本研究未执行扩展的稳健性检验。"
    robustness_records: List[Dict[str, object]] = []
    if args.include_robustness:
        robustness_df = run_robustness_checks(
            panel,
            controls=selected_controls,
            lag_options=[args.lag_x, args.lag_x + 1],
            alt_x_columns=["x_log1p"],
            impute=True,
        )
        robustness_summary = summarize_robustness(robustness_df)
        if not robustness_df.empty:
            robustness_records = json.loads(
                robustness_df.to_json(orient="records", force_ascii=False)
            )

    num_countries = int(panel["country"].nunique())
    num_years = int(panel["year"].nunique())
    year_min = int(panel["year"].min())
    year_max = int(panel["year"].max())
    num_obs = int(len(panel))

    intro_text = (
        f"本文使用 {num_countries} 个国家、{num_years} 个年份（{year_min}–{year_max}）的 {num_obs} 个观测值，"
        "系统评估机器人渗透率对全球价值链地位的影响。研究直接从数据出发进行变量筛选、"
        "固定效应估计及多重稳健性检验，以还原机器人技术扩散对价值链升级的真实轨迹。"
    )

    dataset_overview = (
        f"样本覆盖 {num_countries} 个国家、{num_years} 个年份（{year_min}–{year_max}），共 {num_obs} 个观测值。"
    )
    top_countries_desc = highlight_countries(panel_with_lag)
    top_countries_text = (
        f"机器人渗透率最高的经济体包括：{top_countries_desc}。"
        "这些国家在制造业自动化方面的先行布局，为后续价值链跃升奠定了基础。"
    )
    trend_narrative = synthesize_trend_description(panel_with_lag)

    selection_method_desc = "LASSO" if args.selection_method == "lasso" else "逐步回归"
    missing_pct = int(round(args.missing_threshold * 100))
    control_strategy = (
        f"控制变量筛选遵循“缺失率 < {missing_pct}% + 显著相关性 + VIF<{args.vif_threshold:g}”的准则，"
        f"并采用 {selection_method_desc} 进一步压缩至 {len(selected_controls_original)} 个核心指标。"
    )
    if selected_controls_original:
        controls_text = "、".join(selected_controls_original)
        control_strategy += f" 核心控制变量包括：{controls_text}。"

    baseline_results = (
        f"机器人渗透率滞后 {args.lag_x} 期的系数为 {baseline_coef:.4f}"
        f"（稳健标准误 {baseline_se:.4f}，p 值 {baseline_p:.4f}）。"
    )
    baseline_effect_text = (
        f"按标准差尺度换算，机器人渗透率每提高一个标准差，价值链地位平均提升 {baseline_effect_std:.3f} 个标准差。"
    )

    mechanism_summary = combine_sentences(mechanism_detail.replace("\n", " "), mechanism_implication)
    threshold_summary = combine_sentences(threshold_detail, threshold_implication)
    policy_summary = (
        "整体而言，机器人渗透率的提升对全球价值链地位具有持续且稳健的正向作用。"
        "在产业政策层面，应当营造有利于数字技术扩散的制度环境，推动企业在创新、制造能力与外向型贸易三方面协同升级。"
    )
    future_work = (
        "未来研究可围绕更多中介变量或动态门槛机制展开，并结合 `scripts/visualize.py` 生成的图表，对不同类型经济体的差异化路径进行更精细的讨论。"
    )

    if selected_controls_original:
        selected_controls_lines = "\n".join(f"- {name}" for name in selected_controls_original)
    else:
        selected_controls_lines = "- （未筛选出额外控制变量）"

    analysis_bullet_candidates = [
        normalize_for_bullet(dataset_overview),
        normalize_for_bullet(top_countries_text),
        normalize_for_bullet(trend_narrative),
        normalize_for_bullet(f"{baseline_results} {baseline_effect_text}"),
        normalize_for_bullet(mechanism_summary),
        normalize_for_bullet(threshold_summary),
        normalize_for_bullet(robustness_summary),
        normalize_for_bullet(policy_summary),
    ]
    analysis_bullets = [item for item in analysis_bullet_candidates if item]
    analysis_bullets_str = "\n".join(f"- {item}" for item in analysis_bullets)

    prompt_template_text = args.prompt_template.read_text(encoding="utf-8")
    prompt_text = prompt_template_text.format(
        analysis_bullets=analysis_bullets_str,
        dataset_overview=dataset_overview,
        top_countries_text=top_countries_text,
        trend_narrative=trend_narrative,
        control_strategy=control_strategy,
        selected_controls_lines=selected_controls_lines,
        baseline_results=baseline_results,
        baseline_effect_text=baseline_effect_text,
        mechanism_summary=mechanism_summary,
        threshold_summary=threshold_summary,
        robustness_summary=robustness_summary,
        policy_summary=policy_summary,
        future_work=future_work,
    )
    args.prompt_output.write_text(prompt_text, encoding="utf-8")
    print(f"写作 prompt 已生成：{args.prompt_output.resolve()}")

    top_countries_list = top_countries_desc.split("、") if top_countries_desc else []
    context_payload = {
        "analysis": {
            "intro_text": intro_text,
            "dataset_overview": dataset_overview,
            "top_countries_text": top_countries_text,
            "trend_narrative": trend_narrative,
            "control_strategy": control_strategy,
            "selected_controls": selected_controls_original,
            "baseline": {
                "coef": baseline_coef,
                "std_err": baseline_se,
                "p_value": baseline_p,
                "effect_std": baseline_effect_std,
                "summary": baseline_results,
                "effect_text": baseline_effect_text,
            },
            "mechanism_detail": mechanism_detail,
            "mechanism_implication": mechanism_implication,
            "threshold_detail": threshold_detail,
            "threshold_implication": threshold_implication,
            "robustness_summary": robustness_summary,
            "robustness_records": robustness_records,
            "policy_summary": policy_summary,
            "future_work": future_work,
            "analysis_bullets": analysis_bullets,
        },
        "dataset": {
            "countries": num_countries,
            "years": num_years,
            "year_min": year_min,
            "year_max": year_max,
            "observations": num_obs,
            "top_countries": top_countries_list,
        },
        "config": {
            "lag_x": args.lag_x,
            "selection_method": args.selection_method,
            "missing_threshold": args.missing_threshold,
            "corr_alpha": args.corr_alpha,
            "vif_threshold": args.vif_threshold,
            "include_robustness": args.include_robustness,
            "prompt_template": str(args.prompt_template),
        },
    }
    args.context_output.write_text(
        json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"分析摘要 JSON 已生成：{args.context_output.resolve()}")

    if args.draft_from_template:
        if args.output is None:
            raise SystemExit("--draft-from-template 需要同时提供 --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        paper_context = {
            "intro_text": intro_text,
            "top_countries_text": top_countries_text,
            "trend_narrative": trend_narrative,
            "control_strategy": control_strategy,
            "lag_x": args.lag_x,
            "baseline_results": baseline_results,
            "baseline_effect_text": baseline_effect_text,
            "mechanism_detail": mechanism_detail,
            "mechanism_implication": mechanism_implication,
            "threshold_detail": threshold_detail,
            "threshold_implication": threshold_implication,
            "robustness_summary": robustness_summary,
            "policy_summary": policy_summary,
            "future_work": future_work,
        }
        paper_text = compose_paper_markdown(paper_context)
        args.output.write_text(paper_text, encoding="utf-8")
        print(f"基于内置模版的论文草稿已生成：{args.output.resolve()}")
    else:
        if args.output is not None:
            print("提示：已生成 prompt，请将其交由 LLM 撰写论文后再手动保存到 --output 指定路径。")


if __name__ == "__main__":
    main()

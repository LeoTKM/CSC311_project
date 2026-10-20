#!/usr/bin/env python3
"""
Descriptive statistics for the categorical survey columns of data/class_dataset.csv.

Scope: the three 5-point Likert 单选题 and the three 多选题 (checkbox) columns.
Q1 ("how easy to start", 0-10 numeric scale) is intentionally excluded, and the
five 简答题 (free-text) columns are handled by separate scripts.

Every column is mapped onto the real line:

  单选题 (Likert)    -> ordinal score 1..5
      Strongly disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly agree=5
  多选题 (checkbox)  -> one 0/1 indicator per option (selected=1, not=0)
      plus the derived real feature "number of options selected".

For each mapped feature we report, overall and per label class (equally sized
classes, so unweighted means are comparable): n, mean, sample variance (ddof=1),
sd, the full value distribution per class, a chi-square test of independence,
Cramer's V (effect size) and eta^2 for ordinal/count features.

IMPORTANT: pandas must read this file with keep_default_na=False, otherwise real
answers like "None" / "N/A" (e.g. "which song do you associate with this sport"
-> "None") are silently turned into NaN.

Usage (conda env csc311):
    conda activate csc311
    python analysis/categorical_stats.py                        # TRAIN split (default)
    python analysis/categorical_stats.py --data data/splits/val.csv
    python analysis/categorical_stats.py --data data/splits/train.csv --out-dir analysis/out/train

The default input is the TRAIN split. With the `id`-grouped splits in place
(analysis/make_splits.py), every statistic and every feature must be computed on
train only, otherwise val/test leak into the description of the data. The output
directory defaults to `analysis/out/<file stem>`, so pointing --data at the full
dataset writes to analysis/out/class_dataset and cannot silently overwrite the
train results.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------- mappings ---

# 5-point Likert -> ordinal score (the real-number map used everywhere below).
LIKERT_SCALE = OrderedDict([
    ("Strongly disagree", 1),
    ("Disagree", 2),
    ("Neutral", 3),
    ("Agree", 4),
    ("Strongly agree", 5),
])

# Multi-select vocabularies exactly as they appear in the CSV.
MULTI_VOCAB = OrderedDict([
    ("Q5 场地", [
        "Indoor facility",
        "Outdoor public space",
        "School or university facility",
        "Community recreation facility",
        "Private sports facility",
        "At home or nearby",
        "Others",
    ]),
    ("Q6 季节", ["Winter", "Spring", "Summer", "Fall"]),
    ("Q7 音乐类型", [
        "Pop",
        "Electronic or dance",
        "Latin",
        "Folk or traditional",
        "Classical or jazz",
        "Hip-hop or rap",
        "Country",
        "Rock",
        "No particular type",
        "Other",
    ]),
])

SEASON_INDEX = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}

SINGLE_SPECS = [
    ("Q2", "策略与规划", "Success in this sport depends heavily"),
    ("Q3", "受伤风险", "Players must be comfortable with a high risk of injury"),
    ("Q4", "个人能力胜团队", "Individual brilliance can outweigh teamwork"),
]
MULTI_SPECS = [
    ("Q5", "场地", "Where might you play this sport?"),
    ("Q6", "季节", "During which seasons"),
    ("Q7", "音乐类型", "What types of music do you most associate"),
]

MD = []          # collected markdown lines


def md(line=""):
    MD.append(line)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "na"
    return f"{x:.{nd}f}"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "na"
    return f"{p:.2g}"


# ------------------------------------------------------------------- utils ---

def load(path):
    return pd.read_csv(path, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[])   # keep "None"/"N/A" answers


def col_by_prefix(df, prefix):
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"prefix {prefix!r} matched {len(hits)} columns: {hits}")
    return hits[0]


def describe(values):
    """n, mean, sample variance (ddof=1), sd."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 0, np.nan, np.nan, np.nan
    mu = a.mean()
    var = a.var(ddof=1) if a.size > 1 else np.nan
    return a.size, mu, var, (np.sqrt(var) if var == var else np.nan)


def chi2_with_v(table):
    """table: rows = categories, cols = classes. Returns chi2, dof, p, Cramer's V."""
    table = np.asarray(table, dtype=float)
    if table.shape[0] < 2 or table.shape[1] < 2 or table.sum() == 0:
        return np.nan, 0, np.nan, np.nan
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, dof, p, v


def eta_sq(groups):
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups if len(g)])
    if allv.size < 2:
        return np.nan
    grand = allv.mean()
    ss_tot = ((allv - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_b = sum(len(g) * (np.asarray(g, float).mean() - grand) ** 2 for g in groups if len(g))
    return float(ss_b / ss_tot)


# --------------------------------------------------------------- analysis ----

def analyse_single(df, classes, rows_out):
    md("## 一、单选题（5 点 Likert → 序数 1–5）")
    md()
    md("映射：`Strongly disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly agree=5`。"
       "把李克特标签当等距序数处理，均值可解释为「态度分数」，方差反映意见分歧度。")
    md()

    ranking = []
    for q, cn, prefix in SINGLE_SPECS:
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip()
        score = raw.map(LIKERT_SCALE)
        unexpected = raw[(score.isna()) & (raw != "")]
        if len(unexpected):
            print(f"!! {q}: unmapped values {sorted(set(unexpected))[:5]}")

        valid = pd.DataFrame({"score": score, "label": df["label"]}).dropna()
        n_all, mu, var, sd = describe(valid["score"].values)
        detail = {c: describe(valid.loc[valid.label == c, "score"].values) for c in classes}

        # level x class contingency
        levels = sorted(LIKERT_SCALE.values())
        table = [[int(((valid.label == c) & (valid.score == lv)).sum()) for c in classes]
                 for lv in levels]
        chi2, dof, p, v = chi2_with_v(table)
        eta2 = eta_sq([valid.loc[valid.label == c, "score"].values for c in classes])
        ranking.append((f"{q} {cn}", v, eta2, chi2, p, "single"))

        print(f"\n### {q} {cn}  [single choice -> ordinal 1..5]")
        print(f"    n={n_all} (missing {len(df)-n_all})  mean={fmt(mu)}  var={fmt(var)}  sd={fmt(sd)}")
        print("    " + " | ".join(
            f"{c}: n={detail[c][0]} mean={fmt(detail[c][1])} var={fmt(detail[c][2])}" for c in classes))
        print(f"    chi2={fmt(chi2,2)} dof={dof} p={pfmt(p)}  V={fmt(v)}  eta^2={fmt(eta2)}")

        # ---- markdown block
        md(f"### {q} {cn}")
        md()
        md(f"`{col}`")
        md()
        md("| 指标 | 全体 | " + " | ".join(classes) + " |")
        md("|---|---:|" + "---:|" * len(classes))
        md(f"| n（有效） | {n_all} | " + " | ".join(str(detail[c][0]) for c in classes) + " |")
        md(f"| 均值 | {fmt(mu)} | " + " | ".join(fmt(detail[c][1]) for c in classes) + " |")
        md(f"| 方差 | {fmt(var)} | " + " | ".join(fmt(detail[c][2]) for c in classes) + " |")
        md(f"| 标准差 | {fmt(sd)} | " + " | ".join(fmt(detail[c][3]) for c in classes) + " |")
        md()
        md("取值分布（计数 / 类内占比）：")
        md()
        md("| 序数 | 标签 | " + " | ".join(classes) + " |")
        md("|---|---|" + "---|" * len(classes))
        for i, lv in enumerate(levels):
            lab = [k for k, val in LIKERT_SCALE.items() if val == lv][0]
            cells = []
            for j, c in enumerate(classes):
                cnt = table[i][j]
                cells.append(f"{cnt} ({100*cnt/detail[c][0]:.1f}%)" if detail[c][0] else f"{cnt}")
            md(f"| {lv} | {lab} | " + " | ".join(cells) + " |")
        md()
        md(f"卡方检验：χ²={fmt(chi2,2)}, dof={dof}, p={pfmt(p)}；"
           f"Cramér's V={fmt(v)}（效应量）；η²={fmt(eta2)}（序数分数被类别解释的方差比例）。")
        md()

        for c in classes:
            rows_out.append(dict(question=f"{q} {cn}", kind="single_ordinal", class_=c,
                                 n=detail[c][0], mean=detail[c][1], var=detail[c][2], sd=detail[c][3],
                                 chi2=chi2, dof=dof, p=p, cramers_v=v, eta2=eta2))
        rows_out.append(dict(question=f"{q} {cn}", kind="single_ordinal", class_="ALL",
                             n=n_all, mean=mu, var=var, sd=sd,
                             chi2=chi2, dof=dof, p=p, cramers_v=v, eta2=eta2))

    return ranking


def analyse_multi(df, classes, rows_out, counts_out, ranking):
    md("## 二、多选题（复选框 → 每个选项一个 0/1 变量）")
    md()
    md("映射：某选项被勾选 → 1，未勾选 → 0。因此**均值 = 勾选率（比例）**，"
       "方差 = p(1−p)·n/(n−1)（伯努利方差，p≈0.5 时最大）；另加「勾选个数」这一实数特征。")
    md()

    for (q, cn, prefix), (vname, vocab) in zip(MULTI_SPECS, MULTI_VOCAB.items()):
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip().replace("", np.nan)
        tokens = raw.map(lambda s: {t.strip() for t in s.split(",")} if isinstance(s, str) else set())
        valid_mask = raw.notna()
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue
        labs = df["label"]

        # 1) selection count feature
        cnt = tokens[valid_mask].map(len)
        cnt_by_class = {c: cnt[labs[valid_mask] == c].values for c in classes}
        n_all, mu, var, sd = describe(cnt.values)
        groups = [g for g in cnt_by_class.values() if len(g) > 1]
        F, fp = stats.f_oneway(*groups) if len(groups) == len(classes) else (np.nan, np.nan)
        eta2 = eta_sq(list(cnt_by_class.values()))
        ranking.append((f"{q} {cn} 勾选个数", np.sqrt(eta2), eta2, float(F), float(fp), "count"))

        print(f"\n### {q} {cn}  [multi-select -> 0/1 per option]")
        print(f"    selection_count: n={n_all} mean={fmt(mu)} var={fmt(var)} sd={fmt(sd)}"
              f"  eta^2={fmt(eta2)}  F={(fmt(float(F),2))} p={pfmt(float(fp))}")
        for c in classes:
            n_, m_, v_, _ = describe(cnt_by_class[c])
            print(f"      {c:<10} n={n_:>4} mean={fmt(m_)} var={fmt(v_)}")

        md(f"### {q} {cn}  —— 衍生特征「勾选个数」")
        md()
        md(f"`{col}`")
        md()
        md("| 指标 | 全体 | " + " | ".join(classes) + " |")
        md("|---|---:|" + "---:|" * len(classes))
        md(f"| n（有效） | {n_all} | " + " | ".join(str(describe(cnt_by_class[c])[0]) for c in classes) + " |")
        md(f"| 均值 | {fmt(mu)} | " + " | ".join(fmt(describe(cnt_by_class[c])[1]) for c in classes) + " |")
        md(f"| 方差 | {fmt(var)} | " + " | ".join(fmt(describe(cnt_by_class[c])[2]) for c in classes) + " |")
        md()
        md(f"单因素 ANOVA：F={fmt(float(F),2)}, p={pfmt(float(fp))}；η²={fmt(eta2)}。")
        md()

        # 2) per-option indicators
        md(f"### {q} {cn}  —— 每个选项的 0/1 变量")
        md()
        md("| 选项 | 勾选率 | " + " | ".join(f"{c} 勾选率" for c in classes) +
           " | " + " | ".join(f"{c} 方差" for c in classes) + " | χ² | p | Cramér's V |")
        n_cols = 2 + 2 * len(classes) + 3
        md("|---|" + "---:|" * (n_cols - 1))
        print(f"    {'option':<32}{'rate':>7}  " + "".join(f"{c:>12}" for c in classes) +
              f"{'chi2':>8}{'p':>9}{'V':>7}")
        for opt in vocab:
            ind = tokens[valid_mask].map(lambda s: 1 if opt in s else 0)
            by_class = {c: ind[labs[valid_mask] == c].values for c in classes}
            n_o, mu_o, var_o, sd_o = describe(ind.values)
            table = [[int(sum(by_class[c])), int(len(by_class[c]) - sum(by_class[c]))] for c in classes]
            table = [list(x) for x in zip(*table)]          # rows = selected/not, cols = classes
            chi2, dof, p, v = chi2_with_v(table)
            ranking.append((f"{q} {cn} · {opt}", v, np.nan, chi2, p, "indicator"))

            cells_rate = "".join(f"{100*describe(by_class[c])[1]:>11.1f}%" for c in classes)
            print(f"    {opt[:32]:<32}{100*mu_o:>6.1f}%  {cells_rate}"
                  f"{chi2:>8.2f}{p:>9.3g}{v:>7.3f}")

            md(f"| {opt} | {100*mu_o:.1f}% | " +
               " | ".join(f"{100*describe(by_class[c])[1]:.1f}%" for c in classes) +
               " | " + " | ".join(fmt(describe(by_class[c])[2]) for c in classes) +
               f" | {fmt(chi2,2)} | {pfmt(p)} | {fmt(v)} |")

            rows_out.append(dict(question=f"{q} {cn}", kind="multi_indicator", option=opt, class_="ALL",
                                 n=n_o, rate=mu_o, var=var_o, sd=sd_o, chi2=chi2, dof=dof, p=p, cramers_v=v,
                                 **{f"rate_{c}": describe(by_class[c])[1] for c in classes},
                                 **{f"var_{c}": describe(by_class[c])[2] for c in classes},
                                 **{f"n_{c}": describe(by_class[c])[0] for c in classes}))
        md()
        counts_out.append(dict(question=f"{q} {cn}", feature="selection_count", n=n_all,
                               mean=mu, var=var, sd=sd, F=float(F), p=float(fp), eta2=eta2,
                               **{f"mean_{c}": describe(cnt_by_class[c])[1] for c in classes},
                               **{f"var_{c}": describe(cnt_by_class[c])[2] for c in classes}))

        # 3) season ordinal supplement (assumption-dependent)
        if cn == "季节":
            idx = {c: [] for c in classes}
            for _, r in df.iterrows():
                s = str(r[col]).strip()
                if not s:
                    continue
                for t in {x.strip() for x in s.split(",") if x.strip()}:
                    if t in SEASON_INDEX:
                        idx[r["label"]].append(SEASON_INDEX[t])
            allv = [v for c in classes for v in idx[c]]
            n_s, mu_s, var_s, _ = describe(allv)
            e2s = eta_sq([idx[c] for c in classes])
            print(f"    [supplement] season_index (Winter=1..Fall=4): mean={fmt(mu_s)} var={fmt(var_s)}"
                  f"  eta^2={fmt(e2s)}")
            for c in classes:
                print(f"      {c:<10} mean={fmt(describe(idx[c])[1])} var={fmt(describe(idx[c])[2])}")
            md("补充（假设 Winter=1, Spring=2, Summer=3, Fall=4，按每个被勾选的季节逐项计分）：")
            md()
            md("| 指标 | 全体 | " + " | ".join(classes) + " |")
            md("|---|---:|" + "---:|" * len(classes))
            md(f"| 季节指数均值 | {fmt(mu_s)} | " +
               " | ".join(fmt(describe(idx[c])[1]) for c in classes) + " |")
            md(f"| 季节指数方差 | {fmt(var_s)} | " +
               " | ".join(fmt(describe(idx[c])[2]) for c in classes) + " |")
            md()
            md(f"η²={fmt(e2s)}。该映射依赖「季节可等距排序」这一假设，仅作描述性参考。")
            md()


def write_ranking(ranking):
    md("## 三、按效应量排序（哪些映射特征最能区分 3 个类别）")
    md()
    md("Cramér's V ∈ [0,1]，越大关联越强；η² 为序数/计数特征被类别解释的方差比例。"
       "样本量 870，几乎所有检验都会显著，**请以效应量为准，不要只看 p 值**。")
    md()
    md("| # | 特征 | 类型 | Cramér's V | η² | χ²/F | p |")
    md("|---:|---|---|---:|---:|---:|---:|")
    clean = [r for r in ranking if not np.isnan(r[1])]
    clean.sort(key=lambda r: -r[1])
    for i, (name, v, e2, stat, p, kind) in enumerate(clean, 1):
        md(f"| {i} | {name} | {kind} | {fmt(v)} | {fmt(e2) if e2 == e2 else '—'} | "
           f"{fmt(stat,2)} | {pfmt(p)} |")
    md()

    print("\n=== ranking by Cramer's V ===")
    for i, (name, v, e2, stat, p, kind) in enumerate(clean, 1):
        print(f"{i:>2}. {name:<38} V={fmt(v)}  eta2={fmt(e2) if e2==e2 else '   - '}  p={pfmt(p)}")


def split_section(summary_path, current_file, n_rows_current):
    """Markdown block with the train/val/test sizes and how the split was made."""
    if not os.path.isfile(summary_path):
        return False
    with open(summary_path, encoding="utf-8") as fh:
        s = json.load(fh)

    names = ["train", "val", "test"]
    total_groups = sum(s["splits"][n]["groups"] for n in names)

    md("## 数据切分：train / val / test")
    md()
    md(f"全量数据 `{s['dataset']}`：{s['n_rows']} 行 / {s['n_groups']} 个 `{s['group_col']}`。"
       f"切分清单见 `data/splits/id_to_split.csv` 与 `data/splits/split_summary.json`。")
    md()
    md(f"**本报告只使用 train 集**（`{current_file}`，{n_rows_current} 行）；"
       "val / test 不参与任何统计与画图。")
    md()
    md("| 集合 | 行数 | 分组数（`id`） | 占分组比例 | 各类行数 |")
    md("|---|---:|---:|---:|---|")
    for n in names:
        d = s["splits"][n]
        per_cls = " / ".join(f"{k} {v}" for k, v in d["rows_per_class"].items())
        md(f"| {n} | {d['rows']} | {d['groups']} | {100 * d['groups'] / total_groups:.1f}% | {per_cls} |")
    md(f"| **合计** | **{s['n_rows']}** | **{total_groups}** | **100%** | — |")
    md()
    md("**切分方式**：")
    md()
    md(f"1. 目标比例 **70% / 15% / 15%**（按分组计）。290 个分组无法被 0.15 整除，"
       f"实际切出 train {s['splits']['train']['groups']} 组 / val {s['splits']['val']['groups']} 组 / "
       f"test {s['splits']['test']['groups']} 组（"
       + " / ".join(f"{100 * s['splits'][n]['groups'] / total_groups:.1f}%" for n in names)
       + "）。")
    md(f"2. **按 `{s['group_col']}` 分组切分（group split），不是按行随机切分**："
       f"每个 `{s['group_col']}` 的 {s['splits']['train']['rows_per_group']['min']} 行必须落在同一个集合，"
       "否则同一受访者会同时出现在 train 和 test，指标虚高。")
    md("3. **不是分层抽样（stratified sampling）**，但因为每个 `id` 恰好包含 "
       "Soccer / Baseball / Basketball 各一行，切分后**每个集合天然三类均衡**（1:1:1），效果等价于完美分层。")
    md(f"4. 具体排列由 `sha256(\"<seed>:<id>\")` 排序决定（seed={s['seed']}），"
       "**不依赖任何随机数发生器状态**，因此与 numpy 版本、平台、CSV 行序无关，"
       "重复运行得到的切分文件逐字节一致。")
    md(f"5. 复现：`python analysis/make_splits.py --seed {s['seed']} "
       f"--test-size {s['requested_sizes'].get('test', 0.15)} "
       f"--val-size {s['requested_sizes'].get('val', 0.15)}`，"
       "或用 `--from-manifest data/splits/id_to_split.csv` 直接复用清单。")
    md()
    return True


def main():
    ap = argparse.ArgumentParser(description="single-choice and multi-select descriptive stats")
    ap.add_argument("--data", default="data/splits/train.csv",
                    help="input CSV (default: the train split; never analyse val/test)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: analysis/out/<input file stem>)")
    ap.add_argument("--split-summary", default="data/splits/split_summary.json",
                    help="split manifest used to document train/val/test sizes")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("analysis", "out", Path(args.data).stem)

    df = load(args.data)
    if "label" not in df.columns:
        sys.exit("no label column")
    classes = sorted(df["label"].unique())

    print(f"file: {args.data}   rows: {len(df)}   classes: "
          + ", ".join(f"{c}={int((df.label==c).sum())}" for c in classes))

    md("# 单选 / 多选题：实数映射、均值方差与分类别分布")
    md()
    md(f"数据：`{args.data}`，共 {len(df)} 行 = "
       + " + ".join(str(int((df.label == c).sum())) for c in classes)
       + f" 行（{len(classes)} 类完全均衡：{'、'.join(classes)}）。"
       "三类样本量相同，因此类均值之间可直接比较、无需加权。")
    md()
    md("> 说明：读入时使用 `keep_default_na=False`，避免真实回答 `None` / `N/A` 被误判为缺失。")
    md()

    has_split = split_section(args.split_summary, args.data, len(df))
    if has_split:
        with open(args.split_summary, encoding="utf-8") as fh:
            _s = json.load(fh)
        print("split manifest: " + ", ".join(
            f"{n}={_s['splits'][n]['rows']} rows/{_s['splits'][n]['groups']} ids"
            for n in ("train", "val", "test")))

    if "id" in df.columns:
        n_ids = int(df["id"].nunique())
        per_id = len(df) / n_ids if n_ids else float("nan")
        md(f"> **统计注意事项**：本数据只含 {n_ids} 个 `id`、共 {len(df)} 行，"
           f"平均每个 `id` {per_id:.1f} 行（同一受访者回答了多项运动），行与行并不独立。"
           "下表 χ² / ANOVA 的 p 值是按「行独立」计算的，因此**偏乐观（过于显著）**；"
           "效应量（Cramér's V、η²）是描述性的、不受此影响，"
           "判断特征强弱请以效应量为准，若要严格推断需用聚类稳健方法（如 GEE / 混合效应模型）。")
        md()

    rows_out, counts_out = [], []
    ranking = analyse_single(df, classes, rows_out)
    analyse_multi(df, classes, rows_out, counts_out, ranking)
    write_ranking(ranking)

    os.makedirs(out_dir, exist_ok=True)
    single_df = pd.DataFrame([r for r in rows_out if r["kind"] == "single_ordinal"])
    multi_df = pd.DataFrame([r for r in rows_out if r["kind"] == "multi_indicator"])
    if not single_df.empty:
        single_df.to_csv(os.path.join(out_dir, "single_choice_stats.csv"), index=False)
    if not multi_df.empty:
        multi_df.to_csv(os.path.join(out_dir, "multi_select_stats.csv"), index=False)
    pd.DataFrame(counts_out).to_csv(os.path.join(out_dir, "selection_count_stats.csv"), index=False)
    with open(os.path.join(out_dir, "report_single_multi.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(MD) + "\n")

    print(f"\nwrote {out_dir}/single_choice_stats.csv")
    print(f"wrote {out_dir}/multi_select_stats.csv")
    print(f"wrote {out_dir}/selection_count_stats.csv")
    print(f"wrote {out_dir}/report_single_multi.md")


if __name__ == "__main__":
    main()

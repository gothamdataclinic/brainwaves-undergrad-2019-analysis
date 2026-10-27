"""Pre-specified ERP graph-task analysis (see ERP_ANALYSIS_PLAN.md).

Usage: .venv/bin/python grading/erp_analysis.py
Reads erp_items.csv and erp_llm_raw.jsonl; writes erp_final_grades.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).parent
MODELS = {"anthropic/claude-opus-5-5": "Opus", "openai-codex/gpt-5.6-sol": "GPT"}
QWK_BAR = 0.64  # pre-specified: human benchmark 0.74 minus 0.10


def qwk(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    obs = np.zeros((5, 5))
    for x, y in zip(a, b):
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.subtract.outer(np.arange(5), np.arange(5)) ** 2 / 16
    return 1 - (w * obs).sum() / (w * exp).sum()


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


def ordinal(a, b):
    a, b = np.rint(np.asarray(a, float)).astype(int), np.asarray(b, int)
    return {"n": len(a), "exact %": round(100 * (a == b).mean(), 1),
            "±1 %": round(100 * (abs(a - b) <= 1).mean(), 1), "QWK": round(qwk(a, b), 2)}


def binary(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"n": len(a), "agree %": round(100 * (a == b).mean(), 1), "kappa": round(kappa(a, b), 2)}


def load():
    items = pd.read_csv(HERE / "erp_items.csv", dtype=str, keep_default_na=False)
    items["graph"] = items["graph"].astype(int)
    rows = []
    for line in (HERE / "erp_llm_raw.jsonl").read_text().splitlines():
        rec = json.loads(line)
        for g in rec.get("grades", []):
            rows.append({"model": MODELS.get(rec["model"], rec["model"]), "pass": rec["pass"],
                         "item_id": g["item_id"], "grade": float(g["grade"]),
                         "misc": g["misconception"] == "Yes", "review": int(g.get("flag_review", 0))})
    raw = pd.DataFrame(rows).drop_duplicates(["model", "pass", "item_id"], keep="last")
    raw = raw[raw["pass"].between(1, 3) & raw["model"].isin(MODELS.values())]
    per = raw.groupby(["model", "item_id"]).agg(
        grade=("grade", "median"), votes=("misc", "sum"), passes=("pass", "nunique"),
        review=("review", "max")).reset_index()
    assert per["passes"].eq(3).all(), "every answer needs 3 passes per model"
    per["misc"] = per["votes"] >= 2
    wide = per.pivot(index="item_id", columns="model", values=["grade", "misc", "review"])
    wide.columns = [f"{m}_{v}" for v, m in wide.columns]
    wide = wide.astype({f"{m}_{v}": t for m in MODELS.values()
                        for v, t in [("grade", float), ("misc", bool), ("review", int)]})
    wide["grade"] = wide[[f"{m}_grade" for m in MODELS.values()]].mean(axis=1)
    wide["nonanswer"] = wide[[f"{m}_review" for m in MODELS.values()]].max(axis=1).astype(bool) & wide["grade"].eq(0)
    return items.merge(wide.reset_index(), on="item_id", how="left"), raw


def validation(df, raw):
    """Grader agreement tables and the pre-specified QWK decision."""
    human = df[df["cohort"].eq("Spring") & df["Morgan_Grade"].isin(list("01234"))]
    both = human[human["Rhonda_Grade"].isin(list("01234"))]
    rows = {}
    for m in [*MODELS.values(), "Mean"]:
        col = "grade" if m == "Mean" else f"{m}_grade"
        rows[f"{m} vs Morgan"] = ordinal(human[col], human["Morgan_Grade"].astype(int))
        for v in (1, 2):
            h = human[human["graph"].eq(v)]
            rows[f"{m} vs Morgan, graph v{v}"] = ordinal(h[col], h["Morgan_Grade"].astype(int))
        rows[f"{m} vs Rhonda"] = ordinal(both[col], both["Rhonda_Grade"].astype(int))
    rows["Morgan vs Rhonda (benchmark)"] = ordinal(both["Morgan_Grade"].astype(int), both["Rhonda_Grade"].astype(int))
    graded = df.dropna(subset=["grade"])
    rows["Opus vs GPT (all answers)"] = ordinal(graded["Opus_grade"], np.rint(graded["GPT_grade"]))

    flags = {}
    hm = human[human["Morgan_Misc"].isin(["Yes", "No"])]
    hr = both[both["Rhonda_Misc"].isin(["Yes", "No"])]
    for m in MODELS.values():
        flags[f"{m} vs Morgan"] = binary(hm[f"{m}_misc"], hm["Morgan_Misc"].eq("Yes"))
        flags[f"{m} vs Rhonda"] = binary(hr[f"{m}_misc"], hr["Rhonda_Misc"].eq("Yes"))
    flags["Morgan vs Rhonda (benchmark)"] = binary(hr["Morgan_Misc"].eq("Yes"), hr["Rhonda_Misc"].eq("Yes"))
    flags["Opus vs GPT (all answers)"] = binary(graded["Opus_misc"].astype(bool), graded["GPT_misc"].astype(bool))
    flag_rates = {"Morgan": round(100 * hm["Morgan_Misc"].eq("Yes").mean(), 1),
                  **{m: round(100 * hm[f"{m}_misc"].mean(), 1) for m in MODELS.values()}}

    passes = {}
    for m in MODELS.values():
        p = raw[raw["model"].eq(m)].pivot(index="item_id", columns="pass", values="grade")
        passes[m] = {f"pass {a} v {b} exact %": round(100 * (p[a] == p[b]).mean(), 1)
                     for a, b in [(1, 2), (1, 3), (2, 3)]}

    mean_qwk = rows["Mean vs Morgan"]["QWK"]
    return {
        "grades": pd.DataFrame(rows).T.rename_axis("comparison").reset_index(),
        "flags": pd.DataFrame(flags).T.rename_axis("comparison").reset_index(),
        "flag_rates": flag_rates,
        "passes": pd.DataFrame(passes).T.rename_axis("model").reset_index(),
        "mean_qwk": mean_qwk,
        "confirmatory": mean_qwk >= QWK_BAR,
    }


def paired(df, grade_col, flag_col=None, exclude_nonanswers=False):
    d = df.dropna(subset=[grade_col])
    if exclude_nonanswers:
        d = d[~d["nonanswer"].astype(bool)]
    pre = d[d["phase"].eq("Pre")].set_index("ID")
    post = d[d["phase"].eq("Post")].set_index("ID")
    ids = pre.index.intersection(post.index)
    out = pd.DataFrame({
        "cohort": post.loc[ids, "cohort"], "pre": pre.loc[ids, grade_col], "post": post.loc[ids, grade_col],
        "post_graph": post.loc[ids, "graph"].astype(str),
    })
    out["spring"] = out["cohort"].eq("Spring").astype(int)
    if flag_col:
        out["pre_flag"] = pre.loc[ids, flag_col].astype(float)
        out["post_flag"] = post.loc[ids, flag_col].astype(float)
    return out


def effect(d, label):
    fit = smf.ols("post ~ spring + pre + C(post_graph)", data=d).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["spring"]
    n = d["cohort"].value_counts()
    return {"analysis": label, "n spring": n.get("Spring", 0), "n fall": n.get("Fall", 0),
            "cohort β (spring − fall)": round(fit.params["spring"], 3),
            "95% CI": f"[{lo:.2f}, {hi:.2f}]", "p": round(fit.pvalues["spring"], 4)}


def analysis(df):
    """Descriptives, primary and sensitivity cohort effects, secondary flag model."""
    descriptive = df.dropna(subset=["grade"]).pivot_table(
        index=["cohort", "graph"], columns="phase", values="grade", aggfunc=["mean", "count"]).round(2)
    descriptive.columns = [f"{p} {s}" for s, p in descriptive.columns]

    effects = [effect(paired(df, "grade"), "PRIMARY: two-model mean")]
    for m in MODELS.values():
        effects.append(effect(paired(df, f"{m}_grade"), f"Sensitivity 1: {m} only"))
    effects.append(effect(paired(df, "grade", exclude_nonanswers=True), "Sensitivity 2: non-answers excluded"))

    flags = []
    for m in MODELS.values():
        d = paired(df, f"{m}_grade", flag_col=f"{m}_misc")
        fit = smf.logit("post_flag ~ spring + pre_flag + C(post_graph)", data=d).fit(disp=0)
        rates = d.groupby("cohort")[["pre_flag", "post_flag"]].mean().mul(100).round(1)
        flags.append({"grader": m, "odds ratio (spring/fall)": round(np.exp(fit.params["spring"]), 2),
                      "p": round(fit.pvalues["spring"], 3),
                      **{f"{c} {p.split('_')[0]} %": rates.loc[c, p] for c in rates.index for p in rates.columns}})

    s = df[df["cohort"].eq("Spring") & df["Morgan_Grade"].isin(list("01234"))].assign(
        m=lambda x: x["Morgan_Grade"].astype(int))
    spring_human = []
    for v in (1, 2):
        pre, post = (s[s["graph"].eq(v) & s["phase"].eq(p)]["m"] for p in ("Pre", "Post"))
        spring_human.append({"graph": v, "pre mean": round(pre.mean(), 2), "pre n": len(pre),
                             "post mean": round(post.mean(), 2), "post n": len(post),
                             "Mann–Whitney p": round(stats.mannwhitneyu(pre, post).pvalue, 3)})
    return {
        "descriptive": descriptive.reset_index(),
        "effects": pd.DataFrame(effects),
        "flags": pd.DataFrame(flags),
        "spring_human": pd.DataFrame(spring_human),
    }


if __name__ == "__main__":
    df, raw = load()
    v = validation(df, raw)
    a = analysis(df)
    for name, table in [("Grade agreement", v["grades"]), ("Misconception flag agreement", v["flags"]),
                        ("Pass-to-pass consistency", v["passes"]), ("Descriptive", a["descriptive"]),
                        ("Cohort effect (OLS: post ~ cohort + pre + post graph, HC3)", a["effects"]),
                        ("Secondary: misconception flag at post (logit)", a["flags"]),
                        ("Sensitivity 3: spring, Morgan's grades, pre vs post", a["spring_human"])]:
        print(f"\n== {name} ==\n{table.to_string(index=False)}")
    print(f"\nflag rate, spring human-graded %: {v['flag_rates']}")
    print(f"Pre-specified bar QWK ≥ {QWK_BAR}; two-model mean vs Morgan = {v['mean_qwk']} → "
          + ("confirmatory" if v["confirmatory"] else "exploratory"))
    df.drop(columns=["text"]).to_csv(HERE / "erp_final_grades.csv", index=False)

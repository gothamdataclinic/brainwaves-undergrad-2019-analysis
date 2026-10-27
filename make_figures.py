"""Render Table 1, Table 2, and Figure 1 as Word-ready PNGs.

Reads raw_data/analysis2.xlsx (built by explore.py; fall n=63 after
author-confirmed non-consenting exclusions) and the master workbook for
demographics. Run: .venv/bin/python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parent
OUT = ROOT / "exports"
BLUE, GREY, INK = "#3B5BDB", "#B0B7C3", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": "#888888", "axes.labelcolor": INK})

LABELS = {
    "interpretation": "Interpreting results", "obstacles": "Tolerance for obstacles",
    "readiness": "Readiness for more research", "knowledge": "How knowledge is constructed",
    "process": "Research process in my field", "theory_practice": "Integrating theory & practice",
    "real_problems": "How scientists work on real problems", "evidence": "Assertions require evidence",
    "data_analysis": "Analyzing data", "understanding": "Understanding science",
    "techniques": "Learning lab techniques", "literature": "Reading primary literature",
    "writing": "Science writing", "confidence": "Self-confidence",
    "scientific_thinking": "How scientists think", "independence": "Working independently",
    "community": "Part of a learning community",
}

df = pd.read_excel(ROOT / "raw_data/analysis2.xlsx", sheet_name="self_efficacy")
spring = df[df["cohort"].str.contains("Spring")]
fall = df[df["cohort"].str.contains("Fall")]
is_spring = df["cohort"].str.contains("Spring").astype(int)


def p_fmt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


# ---- item statistics ---------------------------------------------------------
rows = []
for key in ["composite", *LABELS]:
    pre, post = ("Pre", "Post") if key == "composite" else (f"Pre_{key}", f"Post_{key}")
    X = pd.DataFrame({"g": is_spring, "b": df[pre]})
    fit = sm.OLS(df[post], sm.add_constant(X)).fit()
    sg, fg = spring[post] - spring[pre], fall[post] - fall[pre]
    rows.append({"key": key, "item": "Composite (mean of 17 items)" if key == "composite" else LABELS[key],
                 "sD": sg.mean(), "sSD": sg.std(ddof=1), "fD": fg.mean(), "fSD": fg.std(ddof=1),
                 "sSE": sg.sem(), "fSE": fg.sem(), "beta": fit.params["g"], "p": fit.pvalues["g"],
                 "gainp": stats.ttest_ind(sg, fg, equal_var=False).pvalue})
res = pd.DataFrame(rows)
items = res["key"] != "composite"
res.loc[items, "holm"] = multipletests(res.loc[items, "p"], method="holm")[1]

# ---- Table 2 -----------------------------------------------------------------
t2 = [["SURE item", "BrainWaves\nΔ (SD)", "Comparison\nΔ (SD)", "ANCOVA β", "p (Holm)", "Change-\nscore p"]]
for _, r in res.iterrows():
    comp = r["key"] == "composite"
    t2.append([r["item"], f"{r['sD']:+.2f} ({r['sSD']:.2f})", f"{r['fD']:+.2f} ({r['fSD']:.2f})",
               f"{r['beta']:+.2f}", "—" if comp else p_fmt(r["holm"]), p_fmt(r["gainp"])])
fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.axis("off")
tab = ax.table(cellText=t2, loc="center", colWidths=[0.38, 0.15, 0.15, 0.11, 0.10, 0.11])
tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.45)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#D9D9D9"); cell.set_linewidth(0.5)
    cell.get_text().set_ha("left" if c == 0 else "center")
    if r == 0:
        cell.set_facecolor("#EAF0FF"); cell.get_text().set_weight("bold")
    elif r == 1:
        cell.set_facecolor("#F4F6FB"); cell.get_text().set_weight("bold")
    elif res.iloc[r - 1]["holm"] < 0.05 if r > 1 else False:
        cell.set_facecolor("#FFF7E6")
fig.suptitle("Table 2. Self-reported research self-efficacy: change by cohort (SURE, 1–5)", fontsize=10, weight="bold", y=0.98)
fig.text(0.02, 0.015, "Δ = mean post − pre change. ANCOVA: post score ~ cohort + pre score; β = baseline-adjusted BrainWaves advantage.\n"
                      "Holm correction across the 17 items (composite excluded). Change-score p: Welch's t test on Δ, uncorrected.\n"
                      f"BrainWaves spring 2019, n = {len(spring)}; comparison fall 2019, n = {len(fall)}. Highlighted rows: Holm p < .05.",
         fontsize=6.5, color="#555555")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(OUT / "table2_self_efficacy_items.png", dpi=300, facecolor="white")
plt.close(fig)

# ---- Figure 1 ----------------------------------------------------------------
r = res[items].sort_values("holm", ascending=True).iloc[::-1]  # strongest at top after plot
stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
y = np.arange(len(r)); h = 0.38
fig, ax = plt.subplots(figsize=(7.4, 5.8))
ax.barh(y + h / 2, r["sD"], height=h, xerr=r["sSE"], color=BLUE, label=f"BrainWaves (Spring 2019), n = {len(spring)}", error_kw={"lw": 0.8, "capsize": 2})
ax.barh(y - h / 2, r["fD"], height=h, xerr=r["fSE"], color=GREY, label=f"Comparison (Fall 2019), n = {len(fall)}", error_kw={"lw": 0.8, "capsize": 2})
ax.set_yticks(y, [f"{n}  {stars(p)}" if stars(p) else n for n, p in zip(r["item"], r["holm"])], fontsize=7.5)
for tick, p in zip(ax.get_yticklabels(), r["holm"]):
    if p < 0.05: tick.set_weight("bold")
ax.axvline(0, color="#444444", lw=0.9)
ax.set_xlabel("Mean change (post − pre), ± SE", fontsize=8.5)
ax.set_title("Figure 1. Self-reported research self-efficacy: change by item", fontsize=9.5, weight="bold", loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7.5)
fig.text(0.01, 0.005, "SURE items (1–5). Sorted by significance of the baseline-adjusted cohort difference (ANCOVA), Holm-corrected across 17 items: * p<.05, ** p<.01, *** p<.001.",
         fontsize=6.5, color="#555555")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT / "figure1_self_efficacy_items.png", dpi=300, facecolor="white")
plt.close(fig)

# ---- Table 1 -----------------------------------------------------------------
book = ROOT / "raw_data/Undergrad 2019(1).xlsx"
def load_sheet(name):
    t = pd.read_excel(book, sheet_name=name)
    t["ID"] = t["Participant ID"].astype(str).str.strip().str.upper()
    return t
def col(t, prefix): return next(c for c in t.columns if str(c).startswith(prefix))
def race(v):
    if pd.isna(v): return np.nan
    if "," in str(v): return "Multiracial"
    return {1: "White", 2: "Asian", 3: "Hispanic or Latino", 4: "Black or African American",
            5: "Native Hawaiian or Pacific Islander", 6: "Native American or Alaska Native"}[int(v)]
frames = []
for t, name, ids in [(load_sheet("Experimental"), "spring", set(spring["ID"])),
                     (load_sheet("Control"), "fall", set(fall["ID"]))]:
    pre = t[(t["Pre/Post"] == "Pre") & (t["ID"].isin(ids))].drop_duplicates("ID")
    frames.append(pd.DataFrame({
        "cohort": name, "Age": pd.to_numeric(pre[col(t, "What is your age")], errors="coerce"),
        "Gender": pre[col(t, "To which gender")].map({1: "Female", 0: "Male", 2: "Gender variant/non-conforming"}),
        "Class year": pre[col(t, "What is your classification")],
        "Race/ethnicity": pre[col(t, "Race")].map(race),
        "English at home": pre[col(t, "Is English")].map({1: "Yes", 0: "No"}),
        "Prior research": pre[col(t, "Have you participated")].map({1: "Yes", 0: "No"})}))
demo = pd.concat(frames)

def npct(s, level):
    v = s.dropna(); n = int((v == level).sum())
    return f"{n} ({100 * n / len(v):.0f}%)" if len(v) else "—"
t1 = [["Characteristic", f"BrainWaves\nSpring 2019 (n = {len(spring)})", f"Comparison\nFall 2019 (n = {len(fall)})"]]
sp, fa = demo[demo.cohort == "spring"], demo[demo.cohort == "fall"]
t1.append(["Age, mean (SD)", f"{sp['Age'].mean():.1f} ({sp['Age'].std(ddof=1):.1f})", f"{fa['Age'].mean():.1f} ({fa['Age'].std(ddof=1):.1f})"])
for var, levels in [("Gender", ["Female", "Male", "Gender variant/non-conforming"]),
                    ("Class year", ["Freshman/First-Year", "Sophomore", "Junior", "Senior"]),
                    ("Race/ethnicity", ["Asian", "White", "Hispanic or Latino", "Black or African American", "Multiracial"]),
                    ("English at home", ["Yes"]), ("Prior research", ["Yes"])]:
    t1.append([var, "", ""])
    for lv in levels:
        t1.append([f"   {lv}", npct(sp[var], lv), npct(fa[var], lv)])
fig, ax = plt.subplots(figsize=(7.0, 5.6))
ax.axis("off")
tab = ax.table(cellText=t1, loc="center", colWidths=[0.44, 0.28, 0.28])
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.5)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#D9D9D9"); cell.set_linewidth(0.5)
    cell.get_text().set_ha("left" if c == 0 else "center")
    if r == 0: cell.set_facecolor("#EAF0FF"); cell.get_text().set_weight("bold")
    elif t1[r][1] == "": cell.set_facecolor("#F4F6FB"); cell.get_text().set_weight("bold")
fig.suptitle("Table 1. Participant characteristics by cohort", fontsize=10, weight="bold", y=0.98)
fig.text(0.02, 0.015, "Counts are n (% of respondents to that item). Percentages exclude missing responses (≤3 students per item).", fontsize=6.5, color="#555555")
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(OUT / "table1_participants.png", dpi=300, facecolor="white")
plt.close(fig)

print(res[["item", "sD", "fD", "beta", "holm", "gainp"]].round(3).to_string(index=False))
print("Wrote:", *(str(p) for p in sorted(OUT.glob("*.png"))), sep="\n  ")

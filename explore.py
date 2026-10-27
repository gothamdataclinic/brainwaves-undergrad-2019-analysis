import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Dataset lab")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from scipy import stats


@app.function(hide_code=True)
def load_table(path, sheet_name=0):
    """Read one table without modifying sources or matching participants."""
    path = Path(path)
    labels = {}
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(
            path, sheet_name=sheet_name,
            na_values={c: ["-"] for c in ("Pre", "Post", "Rhonda", "Morgan")},
        )
    else:
        frame = pd.read_csv(path)
        # Qualtrics has two metadata RECORDS; question text can span many lines.
        if len(frame) >= 2 and frame.iloc[1].astype("string").str.startswith(
            '{"ImportId":'
        ).fillna(False).all():
            labels = frame.iloc[0].to_dict()
            frame = frame.iloc[2:].reset_index(drop=True)
            for column in frame.filter(regex=r"^Q14_\d+$"):
                # Explicit endpoint labels from these exports; unknown text raises.
                frame[column] = pd.to_numeric(
                    frame[column].replace({"1 (Low)": "1", "5 (High)": "5"}),
                    errors="raise",
                )
    frame.columns = frame.columns.astype(str)
    return frame, labels


@app.cell(hide_code=True)
def _():
    mo.md("""
    # BrainWaves undergraduate study — analysis notebook

    A four-session hands-on EEG module (BrainWaves curriculum + app) was run in
    PSYCH-UA 25 Cognitive Neuroscience (NYU, instructor Pascal Wallisch) in
    **spring 2019**. The **fall 2019** offering (same instructor and curriculum,
    no app sessions) is the **comparison cohort**: nonequivalent, since the
    groups differ by term and were not randomised.

    **Outcomes**

    1. **Self-efficacy (SURE Q14, 17 Likert items)**: self-report, the primary
       outcome. Several hands-on items rise in spring beyond the comparison
       cohort; see Findings for the current numbers.
    2. **ERP graph interpretation (Q20 pre / Q22 post, 0–4 rubric + misconception
       flag)**: the only performance measure. All consented answers, both cohorts
       and both time points, graded by two LLMs validated against the human
       graders (Morgan, Rhonda). Pre-specified plan: `grading/ERP_ANALYSIS_PLAN.md`.

    **Inclusion:** consent = 1 and complete Q14 at pre and post, both cohorts,
    from Pascal's master workbook `Undergrad 2019(1).xlsx`.

    | Section | Purpose |
    |---|---|
    | Interactive explorer | Browse any raw file or the built analysis table |
    | Build `analysis2.xlsx` | Consent, spring + fall self-efficacy, demographics |
    | Self-efficacy results | Composite gains, 17-item table with both tests |
    | Graph task | LLM grader validation, cohort comparison of ERP interpretation |
    | Findings | Top-level summary, limitations, open items |

    All analysis runs locally; raw files are read-only inputs. The notebook is
    plain `explore.py`.

    **Restart:** `.venv/bin/marimo edit explore.py --watch`
    """)
    return


@app.cell(hide_code=True)
def _():
    _root = mo.notebook_location() / "raw_data"
    _files = sorted(p for p in _root.iterdir() if p.suffix.lower() in {".csv", ".xlsx"})
    mo.stop(not _files, mo.md("Add a CSV or XLSX file to `raw_data/`, then rerun this cell."))
    # Default to the built analysis table when it exists, so the explorer opens on
    # the cohort-comparable data rather than a raw single-cohort export.
    _default = "analysis2.xlsx" if (_root / "analysis2.xlsx") in _files else _files[0].name
    dataset = mo.ui.dropdown(
        {p.name: p for p in _files},
        value=_default,
        label="Dataset", searchable=True,
    )
    dataset  # re-scans raw_data/ after file renames
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset):
    if dataset.value.suffix.lower() == ".xlsx":
        with pd.ExcelFile(dataset.value) as _book:
            _sheets = _book.sheet_names
    else:
        _sheets = ["CSV"]
    sheet = mo.ui.dropdown(_sheets, value=_sheets[0], label="Worksheet")
    sheet
    return (sheet,)


@app.cell(hide_code=True)
def _(dataset, sheet):
    df, question_labels = load_table(dataset.value, sheet.value)
    mo.stop(df.empty, mo.md("This table has no data rows. Choose another sheet or file."))
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    mo.md(f"**{len(df):,} rows · {len(df.columns)} columns · {len(numeric_columns)} numeric columns**")
    return df, numeric_columns, question_labels


@app.cell(hide_code=True)
def _(df, question_labels):
    _profile = pd.DataFrame({
        "column": df.columns,
        "type": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "unique": df.nunique().values,
        "question": [question_labels.get(c, c) for c in df.columns],
    })
    mo.accordion({
        "Data and column dictionary": mo.vstack([mo.ui.table(df), mo.ui.table(_profile)]),
        "Numeric summary": mo.ui.table(df.describe().T.reset_index()),
        "Import notes and privacy": mo.md("""
            Qualtrics question labels and ImportId records are metadata, not respondents.
            Only `Q14_*` scores are decoded: `1 (Low)` → 1 and `5 (High)` → 5;
            blanks remain missing. All response rows are retained, including incomplete
            responses and previews. Set inclusion criteria explicitly before inference.

            Files are explored separately: duplicate exports are not combined, and no
            cohort assignment, ID matching, or imputation is performed. Excel uses the
            first row as headers; pandas suffixes duplicate names, e.g. `Misconception.1`.
            In the Excel score columns `Pre`, `Post`, `Rhonda`, and `Morgan`, `-` is
            treated as missing, not zero. The column dictionary reports missing counts.
            IDs and numeric category codes are not automatically meaningful measures.
            Survey items are ordinal; smooth density curves are only exploratory.

            Raw tables may contain student identifiers and IP addresses. Keep the server
            on localhost. Review tables and hover fields before exporting or sharing;
            do not send identifiable rows to an external AI service.
        """),
    })
    return


@app.cell(hide_code=True)
def _(df, numeric_columns):
    # Sensible defaults: prefer composite/gain columns, then item-level scores.
    # Excludes ID-like columns, which are labels rather than measures.
    _id_like = {"ID", "StudentID", "Student ID", "Subject", "cohort", "grader", "Version"}
    _measures = [c for c in numeric_columns if c not in _id_like]
    _priority = [c for c in ("Gain", "Pre", "Post") if c in _measures]
    _items = [c for c in _measures if c.startswith(("Pre_", "Post_", "Q14_"))]
    _ordered = _priority + [c for c in _items if c not in _priority]
    _ordered = _ordered or _measures

    _x_default = _ordered[0] if _ordered else (df.columns[0] if len(df.columns) else None)
    _y_default = _ordered[1] if len(_ordered) > 1 else _x_default
    # Default grouping to cohort when present, so cohorts are visible immediately.
    _group_default = "cohort" if "cohort" in df.columns else None

    chart_type = mo.ui.dropdown(
        ["Histogram", "Box", "Violin", "Scatter", "ECDF", "Correlation heatmap"],
        value="Box" if _group_default else "Histogram", label="Chart",
    )
    # X offers measures only; grouping/faceting is where categorical columns belong.
    x = mo.ui.dropdown(_measures or df.columns.tolist(), value=_x_default,
                       label="Measure / X", searchable=True)
    y = mo.ui.dropdown(_measures, value=_y_default, label="Compare with / Y", searchable=True)
    group = mo.ui.dropdown(
        [c for c in df.columns if df[c].nunique(dropna=True) <= 12],
        value=_group_default, label="Group / color (optional)", searchable=True,
    )
    heatmap_columns = mo.ui.multiselect(
        _measures, value=(_ordered[:8] or _measures), label="Heatmap columns",
    )
    mo.vstack([
        mo.md("""
            ## Interactive explorer
            Pick any file in `raw_data/` above. For `analysis2.xlsx`, the
            `self_efficacy` sheet is the analysis table: one row per participant,
            `Pre`/`Post`/`Gain` composites plus the 17 named Q14 items.

            **X** is the measure. **Y** is used by Scatter and by the tests below.
            **Group** splits by a categorical column (defaults to `cohort`).
            Only low-cardinality columns are offered as groups, and ID columns are
            excluded from the measure lists because they are labels, not scores.
        """),
        mo.hstack([chart_type, x, y, group], justify="start", wrap=True),
        mo.accordion({"Choose heatmap columns": heatmap_columns}),
    ])
    return chart_type, group, heatmap_columns, x, y


@app.cell(hide_code=True)
def _(
    chart_type,
    df,
    group,
    heatmap_columns,
    numeric_columns,
    question_labels,
    x,
    y,
):
    _kind = chart_type.value
    if _kind == "Correlation heatmap":
        mo.stop(len(heatmap_columns.value) < 2, mo.md("Choose at least two heatmap columns."))
        _correlation = df[heatmap_columns.value].corr(method="spearman", min_periods=3)
        _figure = px.imshow(
            _correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
            title="Spearman correlation · pairwise complete observations",
        )
    elif _kind == "Histogram":
        _figure = px.histogram(df, x=x.value, color=group.value, barmode="overlay")
    else:
        mo.stop(x.value not in numeric_columns, mo.md("Choose a numeric X for this chart."))
        if _kind == "Box":
            _figure = px.box(df, y=x.value, x=group.value, color=group.value, points="all")
        elif _kind == "Violin":
            _figure = px.violin(df, y=x.value, x=group.value, color=group.value, box=True, points="all")
        elif _kind == "ECDF":
            _figure = px.ecdf(df, x=x.value, color=group.value)
        else:
            mo.stop(y.value is None, mo.md("Choose a numeric Y for a scatter plot."))
            _figure = px.scatter(df, x=x.value, y=y.value, color=group.value, opacity=0.65)
    _figure.update_layout(template="plotly_white", height=500, margin=dict(l=40, r=20, t=50, b=40))
    mo.vstack([
        mo.md(f"**X:** {question_labels.get(x.value, x.value)}  \n**Y:** {question_labels.get(y.value, y.value)}"),
        mo.ui.plotly(_figure),
    ])
    return


@app.cell(hide_code=True)
def _():
    test = mo.ui.dropdown(
        ["Spearman correlation", "Pearson correlation", "Paired t-test", "Wilcoxon signed-rank"],
        value="Spearman correlation", label="Statistical test",
    )
    confirm_design = mo.ui.checkbox(
        label="I confirm rows are independent units and X/Y belong to the same unit."
    )
    mo.vstack([
        mo.md("""
            ## Statistical sandbox
            Uses X and Y above, dropping rows missing either value or containing infinity.
            For pre/post tests, each row **must already be one matched participant**;
            never match separate exports by row order.

            **Spearman:** monotonic association, suitable for ordinal scores; its p-value
            is asymptotic (small/tied samples warrant a permutation analysis).
            **Pearson:** linear association; classical inference assumes bivariate normality.
            **Paired t:** mean Y−X; inspect differences for outliers and non-normality.
            **Wilcoxon:** paired differences with an approximately symmetric distribution.
            All p-values below are two-sided and **unadjusted**; trying many tests is
            exploration, not confirmatory evidence. These scores alone do not establish causality.
        """),
        mo.hstack([test, confirm_design], justify="start", wrap=True),
    ])
    return confirm_design, test


@app.cell(hide_code=True)
def _(confirm_design, df, numeric_columns, test, x, y):
    mo.stop(not confirm_design.value, mo.md("Confirm the sampling/matching design to run a test."))
    mo.stop(
        x.value not in numeric_columns or y.value not in numeric_columns or x.value == y.value,
        mo.md("Choose two different numeric columns for X and Y."),
    )
    _pairs = df[[x.value, y.value]].replace([float("inf"), -float("inf")], float("nan")).dropna()
    mo.stop(len(_pairs) < 3, mo.md("Need at least three complete, finite pairs."))
    _a = _pairs[x.value].to_numpy(dtype=float)
    _b = _pairs[y.value].to_numpy(dtype=float)
    _kind = test.value
    if _kind.endswith("correlation"):
        mo.stop((_pairs.nunique() < 2).any(), mo.md("Correlation is undefined for a constant column."))
        _result = stats.spearmanr(_a, _b) if _kind.startswith("Spearman") else stats.pearsonr(_a, _b)
    elif _kind == "Paired t-test":
        mo.stop((_b - _a).std() == 0, mo.md("The t-test is undefined when differences have zero variance."))
        _result = stats.ttest_rel(_b, _a)
    else:
        mo.stop(not (_b != _a).any(), mo.md("All paired differences are zero; no signed-rank test is reported."))
        _result = stats.wilcoxon(_b - _a)
    _summary = {
        "test": _kind, "X": x.value, "Y": y.value, "complete pairs": len(_pairs),
        "excluded rows": len(df) - len(_pairs),
        "statistic": float(_result.statistic), "p (unadjusted)": float(_result.pvalue),
    }
    if _kind == "Paired t-test":
        _ci = _result.confidence_interval()
        _summary.update({
            "mean Y - X": float((_b - _a).mean()),
            "95% CI low": float(_ci.low), "95% CI high": float(_ci.high),
        })
    mo.ui.table(pd.DataFrame([_summary]), selection=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Keep exploring in Python
    Add a cell using `df`, `px`, and `stats`. Ask for a different encoding, a facet,
    a missingness plot, or a participant-level pre/post view; the code stays visible.
    SciPy also includes Welch's t-test (`stats.ttest_ind(..., equal_var=False)`),
    Mann–Whitney, Fisher's exact test, bootstrap intervals, and permutation tests.
    Independent-sample tests require independent groups, not repeated observations.

    **Keep the stack small:** Plotly covers the interactive chart families here.
    Add seaborn for static publication styling or statsmodels for regression/mixed models
    when a specific analysis needs them. No cloud accounts or API keys are required.

    ---

    The sections below build the analysis table and run the confirmatory tests
    that the Findings section summarises.
    """)
    return


@app.cell(hide_code=True)
def _(fall_comparison, spring_augmented):
    # ── Step 3: Combine into the final analysis table ───────────────────
    # Both cohorts: Q14 composite (mean of 17 items, 1–5), matched pre/post,
    # consented students only.

    augmented = pd.concat([spring_augmented, fall_comparison], ignore_index=True)
    augmented["Gain"] = augmented["Post"] - augmented["Pre"]

    # Summary counts
    _spring_n = (augmented["cohort"] == "Spring (brainwaves)").sum()
    _fall_n = (augmented["cohort"] == "Fall (comparison)").sum()

    # Save to disk
    _out_path = mo.notebook_location() / "raw_data" / "analysis2.xlsx"
    augmented.to_excel(_out_path, index=False)

    # De-identified copy for data sharing: no IDs, rows shuffled so the
    # participant number does not follow ID order.
    _deid = augmented.drop(columns="ID").sample(frac=1, random_state=0)
    _deid.insert(0, "participant", range(1, len(_deid) + 1))
    _export = mo.notebook_location() / "exports" / "self_efficacy_deidentified.csv"
    _export.parent.mkdir(exist_ok=True)
    _deid.to_csv(_export, index=False)

    mo.vstack([
        mo.md(f"""
    ### Augmented analysis table — saved to `raw_data/analysis2.xlsx`

    | | Spring (brainwaves) | Fall (comparison) |
    |---|---|---|
    | **N** | {_spring_n} | {_fall_n} |
    | **Pre scores** | Q14 mean (1–5 Likert) | Q14 mean (1–5 Likert) |
    | **Post scores** | Q14 mean (1–5 Likert) | Q14 mean (1–5 Likert) |
    | **Measure** | SURE self-efficacy composite | SURE self-efficacy composite |
    | **Manipulation** | Brainwaves EEG module | None |
    | **Gain** | Post − Pre | Post − Pre |

    **Columns:** `ID`, `cohort`, `Pre`, `Post`, `Gain`

    **Both cohorts are now on the same 1–5 Likert scale** (Q14 composite).
    Direct comparison of gains is valid.
    """),
        mo.ui.table(augmented, selection=None),
    ])
    return (augmented,)


@app.cell(hide_code=True)
def _(augmented):
    # ── Gain comparison: Spring (brainwaves) vs Fall (comparison) ───────────
    _spring = augmented[augmented["cohort"] == "Spring (brainwaves)"]["Gain"].dropna()
    _fall = augmented[augmented["cohort"] == "Fall (comparison)"]["Gain"].dropna()

    # Both cohorts now on the same 1–5 Likert scale — direct comparison is valid
    _spring_d = _spring.mean() / _spring.std() if _spring.std() > 0 else 0
    _fall_d = _fall.mean() / _fall.std() if _fall.std() > 0 else 0

    # Welch's t-test on gains
    _welch = stats.ttest_ind(_spring, _fall, equal_var=False)
    # Mann-Whitney as a non-parametric alternative
    _mw = stats.mannwhitneyu(_spring, _fall, alternative="two-sided")

    # Bar chart: mean gain ± SE by cohort
    _gain_summary = augmented.dropna(subset=["Gain"]).groupby("cohort")["Gain"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    _gain_summary["se"] = _gain_summary["std"] / np.sqrt(_gain_summary["count"])

    _fig = px.bar(
        _gain_summary,
        x="cohort", y="mean",
        error_y="se",
        color="cohort",
        color_discrete_map={
            "Spring (brainwaves)": "#636EFA",
            "Fall (comparison)": "#EF553B",
        },
        labels={"mean": "Mean Gain (Post − Pre)", "cohort": ""},
        title="Q14 Self-Efficacy: Pre → Post Gain by Cohort",
    )
    _fig.update_layout(
        template="plotly_white", height=450,
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    _fig.add_hline(y=0, line_dash="dot", line_color="gray")

    mo.vstack([
        mo.md(f"""
    ### Q14 self-efficacy gain comparison (same 1–5 scale)

    | | Spring (brainwaves) | Fall (comparison) |
    |---|---|---|
    | **N** | {len(_spring)} | {len(_fall)} |
    | **Mean Gain** | {_spring.mean():.3f} | {_fall.mean():.3f} |
    | **SD** | {_spring.std():.3f} | {_fall.std():.3f} |
    | **Cohen's d (within)** | {_spring_d:.3f} | {_fall_d:.3f} |
    | **Welch t** | t = {_welch.statistic:.3f}, p = {_welch.pvalue:.4f} |
    | **Mann-Whitney U** | U = {_mw.statistic:.1f}, p = {_mw.pvalue:.4f} |

    Both cohorts use the Q14 SURE self-efficacy composite (mean of 17 Likert
    items, 1–5 scale). Gains are directly comparable.
    """),
        mo.ui.plotly(_fig),
    ])
    return


@app.cell
def _(augmented, q14_labels):
    # ── Item-level results: all 17 items, both between-cohort tests ──────
    # Paired p: within-cohort change against zero. Between cohorts, the
    # gain-score test (Welch on Post − Pre) and ANCOVA (Post ~ group + Pre) are
    # both reported; they answer different questions when baselines differ
    # (Lord's paradox). Holm correction across the 17 items, per test.
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    _is_spring = augmented["cohort"].eq("Spring (brainwaves)").astype(int)
    _rows = []
    for _item in [*q14_labels.values(), "composite"]:
        _pre, _post = (
            ("Pre", "Post") if _item == "composite" else (f"Pre_{_item}", f"Post_{_item}")
        )
        _d = augmented[[_pre, _post]].assign(group=_is_spring).dropna()
        _gain = _d[_post] - _d[_pre]
        _sg, _fg = _gain[_d["group"] == 1], _gain[_d["group"] == 0]
        _ancova = sm.OLS(_d[_post], sm.add_constant(_d[["group", _pre]])).fit()
        _rows.append({
            "item": _item,
            "spring Δ": _sg.mean(),
            "spring paired p": stats.ttest_1samp(_sg, 0).pvalue,
            "comparison Δ": _fg.mean(),
            "comparison paired p": stats.ttest_1samp(_fg, 0).pvalue,
            "gain-score p": stats.ttest_ind(_sg, _fg, equal_var=False).pvalue,
            "ANCOVA β": _ancova.params["group"],
            "ANCOVA p": _ancova.pvalues["group"],
        })
    item_results = pd.DataFrame(_rows)
    _items = item_results["item"].ne("composite")
    for _col in ["gain-score p", "ANCOVA p"]:
        item_results.loc[_items, f"{_col} (Holm)"] = multipletests(
            item_results.loc[_items, _col], method="holm"
        )[1]
    item_results = item_results.round(4)

    mo.vstack([
        mo.md("""
    ### Item-level results: all 17 SURE items

    Δ = mean Post − Pre. Paired p tests each cohort's change against zero.
    Between cohorts: **gain-score p** (Welch on Δ) and **ANCOVA** (Post ~ group +
    Pre; β = baseline-adjusted spring advantage). Holm columns correct across the
    17 items; the composite row is not part of the correction.
    """),
        mo.ui.table(item_results, selection=None),
    ])
    return (item_results,)


@app.cell(hide_code=True)
def _(augmented, item_results):
    # ── Figure: pre→post change per item, asterisks = Holm ANCOVA ────────
    import plotly.graph_objects as go

    _text = {
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
    _r = item_results[item_results["item"].ne("composite")].set_index("item")
    _order = _r.sort_values("ANCOVA p (Holm)", ascending=False).index  # bottom → top

    def _label(item):
        _p = _r.loc[item, "ANCOVA p (Holm)"]
        _stars = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else ""
        return f"<b>{_text[item]}  {_stars}</b>" if _stars else _text[item]

    _item_figure = go.Figure()
    for _cohort, _name, _color in [
        ("Fall (comparison)", "Comparison (Fall 2019)", "#B0B7C3"),
        ("Spring (brainwaves)", "BrainWaves (Spring 2019)", "#3B5BDB"),
    ]:
        _d = augmented[augmented["cohort"].eq(_cohort)]
        _gain = pd.DataFrame({_i: _d[f"Post_{_i}"] - _d[f"Pre_{_i}"] for _i in _order})
        _item_figure.add_bar(
            y=[_label(_i) for _i in _order], x=_gain.mean(), orientation="h",
            name=f"{_name}, n={len(_d)}", marker_color=_color,
            error_x=dict(type="data", array=_gain.sem(), thickness=1, width=3),
        )
    _item_figure.add_vline(x=0, line_color="#444", line_width=1)
    _item_figure.update_layout(
        template="plotly_white", barmode="group", width=1100, height=840, bargap=0.25,
        font=dict(size=14), xaxis_title="Mean change (post − pre)",
        legend=dict(orientation="h", y=-0.08, x=0), margin=dict(l=20, r=30, t=120, b=90),
        title=dict(x=0.01, text=(
            "<b>Self-reported research self-efficacy: pre→post change by item</b><br>"
            "<sup>SURE items, 1–5 Likert; mean change ± SE; consented students; sorted by "
            "significance of the cohort difference.<br>Asterisks: baseline-adjusted cohort "
            "difference (ANCOVA), Holm-corrected across 17 items. * p<.05, ** p<.01, *** p<.001</sup>"
        )),
    )
    _out = mo.notebook_location() / "exports" / "self_efficacy_items.html"
    _out.parent.mkdir(exist_ok=True)
    _item_figure.write_html(_out, include_plotlyjs=True)
    mo.ui.plotly(_item_figure)
    return


@app.cell(hide_code=True)
def _(augmented, erp_results, erp_validation, item_results):
    # ── Findings: numbers computed live from the analysis table ─────────
    _composite = pd.DataFrame([
        {
            "cohort": _cohort,
            "N": len(_d),
            "Pre": round(_d["Pre"].mean(), 2),
            "Post": round(_d["Post"].mean(), 2),
            "Gain": round(_d["Gain"].mean(), 3),
            "paired p": round(stats.ttest_1samp(_d["Gain"], 0).pvalue, 4),
        }
        for _cohort, _d in augmented.groupby("cohort", sort=False)
    ])
    _c = item_results[item_results["item"].eq("composite")].iloc[0]
    _headline = item_results[item_results["ANCOVA p (Holm)"] < 0.05][[
        "item", "spring Δ", "comparison Δ", "comparison paired p",
        "ANCOVA p", "ANCOVA p (Holm)", "gain-score p",
    ]]
    _is_spring = augmented["cohort"].eq("Spring (brainwaves)")
    _baseline = stats.ttest_ind(
        augmented.loc[_is_spring, "Pre"], augmented.loc[~_is_spring, "Pre"], equal_var=False
    )
    _erp = erp_results["effects"].iloc[0]
    mo.vstack([
        mo.md("""
    ## Findings

    ### 1. Self-efficacy (primary outcome)

    SURE composite (mean of 17 items, 1–5); consented students with complete
    pre and post Q14:
    """),
        mo.ui.table(_composite, selection=None),
        mo.md(f"""
    Composite between cohorts: gain-score p = {_c["gain-score p"]:.4f};
    ANCOVA β = {_c["ANCOVA β"]:.3f}, p = {_c["ANCOVA p"]:.4f}. Baseline composite:
    spring {augmented.loc[_is_spring, "Pre"].mean():.2f} vs comparison
    {augmented.loc[~_is_spring, "Pre"].mean():.2f} (Welch p = {_baseline.pvalue:.3f}).
    When baselines differ the two tests can disagree (Lord's paradox): the
    gain-score test assumes each cohort would otherwise have stayed at its own
    level; ANCOVA assumes both would drift toward a common mean. With the same
    lecturer, course, and curriculum, ANCOVA's assumption is plausible.
    **Report both.**

    **Items passing Holm-corrected ANCOVA** (the headline claim). Where the
    uncorrected gain-score p is ≥ .05, the tests disagree; say so in one sentence.
    """),
        mo.ui.table(_headline, selection=None),
        mo.md(f"""
    All other items, including scientific thinking, appear only in the full
    17-item table.

    ### 2. ERP graph task (performance measure)

    Graded by Claude Opus 5.5 and GPT-5.6 (3 passes each), validated on spring
    answers whose human grades the graders never saw: two-model mean vs Morgan
    QWK = **{erp_validation["mean_qwk"]:.2f}**, above the pre-specified bar (0.64)
    and the human benchmark (Morgan vs Rhonda 0.74), so the comparison is
    confirmatory.

    **No cohort difference.** Post grade adjusted for pre grade and graph
    version: spring − fall = {_erp["cohort β (spring − fall)"]:+.2f}
    {_erp["95% CI"]}, p = {_erp["p"]:.2f} (n = {_erp["n spring"]} spring,
    {_erp["n fall"]} fall). Every sensitivity analysis agrees. The misconception
    flag is secondary and less reliable (model–human kappa 0.25–0.34 vs human
    0.44); it shows no cohort difference either.

    Both cohorts were taught the same ERP content in lecture, so this fits the
    paper's story: the program raised confidence doing research, not content
    knowledge.

    ### 3. Claim wording

    Write "students reported," not "students learned." Credit the **BrainWaves
    program** (curriculum + app). Interpretation: both cohorts received the same
    ERP content, so the program built confidence doing research rather than new
    content knowledge.

    ### Limitations

    - Nonequivalent cohorts (different terms, no randomisation); the two
      between-cohort tests disagree on the composite.
    - Cohorts differ in class year (spring mostly juniors; fall spread across
      sophomores, juniors, seniors). See the Participants table.
    - Self-report outcome.
    - One course and lecturer (Pascal Wallisch) at one institution. Spring lab
      sessions were taught by the TAs (Morgan Glover, Rhonda Kolaric, Laura,
      Omri); who led the fall lab sections is not yet confirmed.
    - The app's effect can't be separated from facilitation (app training, TA
      delivery).
    - ERP graph task graded by LLMs, not humans (validated against the human
      graders; disclose in the supplement). The misconception flag is only
      moderately reliable and is not used for claims.

    ### Open items

    1. Fall lab staffing: same TAs as spring?
    2. Reconcile the instructor's 63-student fall set with the master workbook
       (three records; student IDs withheld from this public copy).
    3. One fall pre record: the workbook and the instructor's fall export disagree, and no raw
       fall pre-survey export exists to settle it (workbook value used).
    4. IRB (NYU IRB-FY2017-1171, Steven Azeka; approval on file dated
       2019-07-30): confirm the approval in force during spring 2019 data
       collection (Apr–May 2019); the 2019-07-30 date is presumably a renewal.
    5. IRB: earlier exploratory fall LLM runs included text from non-consented
       students (the validated runs used consented text only). Raise with Steve.
    6. IRB: permission to share de-identified data.
    """),
    ])
    return


@app.cell(hide_code=True)
def _():
    # ── Cohorts: Pascal's master workbook is the single source ──────────
    # Undergrad 2019(1).xlsx holds consent (1 = yes), Q14 and demographics for
    # both cohorts with student IDs already corrected. Its consent codes match
    # the fall roster exactly. Inclusion: consent = 1 and complete Q14 at pre
    # and post. Item columns are in Qualtrics Q14_1..Q14_17 order.
    _root = mo.notebook_location() / "raw_data"
    _book = _root / "Undergrad 2019(1).xlsx"
    q14_labels = {
        "Q14_1": "interpretation",
        "Q14_2": "obstacles",
        "Q14_3": "readiness",
        "Q14_4": "knowledge",
        "Q14_5": "process",
        "Q14_6": "theory_practice",
        "Q14_7": "real_problems",
        "Q14_8": "evidence",
        "Q14_9": "data_analysis",
        "Q14_10": "understanding",
        "Q14_11": "techniques",
        "Q14_12": "literature",
        "Q14_13": "writing",
        "Q14_14": "confidence",
        "Q14_15": "scientific_thinking",
        "Q14_16": "independence",
        "Q14_17": "community",
    }
    _q14 = list(q14_labels)

    def _sheet(name):
        _t = pd.read_excel(_book, sheet_name=name)
        _t = _t.rename(columns=dict(zip(_t.columns[5:22], _q14)))
        _t["ID"] = _t["Participant ID"].astype(str).str.strip().str.upper()
        return _t

    _overview, _exp, _ctl = (_sheet(_n) for _n in ("Overview", "Experimental", "Control"))
    # Fall consent is blank on Overview; the Control sheet carries it.
    spring_consented = set(
        _overview.loc[_overview["Group"].eq("Experimental") & _overview["Consent"].eq(1), "ID"]
    )
    fall_consented = set(_ctl.loc[_ctl["Consent"].eq(1), "ID"])

    # For a few students the workbook keeps an unfinished retake submitted a
    # week later. Rule: first complete submission, from the raw exports. (No
    # raw fall pre-survey export exists, so fall pre uses the workbook.)
    _first = {}
    for _group, _phase, _files in [
        ("Experimental", "Pre", ["Prev1.csv", "Prev2 (1).csv"]),
        ("Experimental", "Post", ["Postv1.csv", "Postv2.csv"]),
        ("Control", "Post", ["Fall Post v1.csv", "Fall Post v2.csv"]),
    ]:
        _t = pd.concat([load_table(_root / _f)[0] for _f in _files], ignore_index=True)
        _t["ID"] = _t["Q9"].astype(str).str.strip().str.upper()
        _t = _t[_t["ID"].duplicated(keep=False) & _t["Finished"].astype(str).eq("True")]
        for _, _r in _t.sort_values("RecordedDate").drop_duplicates("ID").iterrows():
            _first[(_group, _r["ID"], _phase)] = _r[_q14].to_numpy(dtype=float)

    def _cohort(t, group, name):
        _t = t[t["Consent"].eq(1)].dropna(subset=_q14)
        _t = _t[_t.groupby("ID")["Pre/Post"].transform("nunique").eq(2)]
        _rows = []
        for _id, _g in _t.groupby("ID"):
            _row = {"ID": _id, "cohort": name}
            for _phase in ("Pre", "Post"):
                _v = _first.get(
                    (group, _id, _phase),
                    _g.loc[_g["Pre/Post"].eq(_phase), _q14].iloc[0].to_numpy(dtype=float),
                )
                _row.update({f"{_phase}_{q14_labels[_c]}": _x for _c, _x in zip(_q14, _v)})
            _rows.append(_row)
        _df = pd.DataFrame(_rows)
        _df["Pre"] = _df.filter(regex="^Pre_").mean(axis=1)
        _df["Post"] = _df.filter(regex="^Post_").mean(axis=1)
        return _df

    spring_augmented = _cohort(_exp, "Experimental", "Spring (brainwaves)")
    fall_comparison = _cohort(_ctl, "Control", "Fall (comparison)")
    # Author-confirmed non-consenting fall records, despite workbook consent
    # flags. Their IDs are withheld from this public copy and are supplied
    # locally in raw_data/excluded_fall_ids.txt (never published).
    _excl = _root / "excluded_fall_ids.txt"
    _excluded = set(_excl.read_text().split()) if _excl.exists() else set()
    fall_comparison = fall_comparison[~fall_comparison["ID"].isin(_excluded)]

    # Demographics from each analysed student's pre-survey row (numeric codes
    # as stored in the workbook).
    def _col(t, prefix):
        return next(_c for _c in t.columns if str(_c).startswith(prefix))

    def _race(v):
        if pd.isna(v):
            return np.nan
        if "," in str(v):
            return "Multiracial"
        return {1: "White", 2: "Asian", 3: "Hispanic or Latino", 4: "Black or African American",
                5: "Native Hawaiian or Pacific Islander", 6: "Native American or Alaska Native"}[int(v)]

    _yes_no = {1: "Yes", 0: "No"}
    demographics = pd.concat([
        pd.DataFrame({
            "ID": _pre["ID"],
            "cohort": _name,
            "Age": pd.to_numeric(_pre[_col(_t, "What is your age")], errors="coerce"),
            "Gender": _pre[_col(_t, "To which gender")].map({1: "Female", 0: "Male", 2: "Gender variant/non-conforming"}),
            "Class year": _pre[_col(_t, "What is your classification")],
            "Race/ethnicity": _pre[_col(_t, "Race")].map(_race),
            "English main language at home": _pre[_col(_t, "Is English")].map(_yes_no),
            "Prior research experience": _pre[_col(_t, "Have you participated")].map(_yes_no),
        })
        for _t, _name, _ids in [
            (_exp, "Spring (brainwaves)", spring_augmented["ID"]),
            (_ctl, "Fall (comparison)", fall_comparison["ID"]),
        ]
        for _pre in [_t[_t["Pre/Post"].eq("Pre") & _t["ID"].isin(_ids)].drop_duplicates("ID")]
    ], ignore_index=True)

    mo.md(f"""
    ### Cohorts (`Undergrad 2019(1).xlsx`)

    | | Spring (BrainWaves) | Fall (comparison) |
    |---|---:|---:|
    | Consented | {len(spring_consented)} | {len(fall_consented)} |
    | Consented + complete pre and post Q14 | **{len(spring_augmented)}** | **{len(fall_comparison)}** |

    First-complete-submission overrides applied: {len(_first)} (student, survey) pairs
    with duplicate raw submissions.
    """)
    return (
        demographics,
        fall_comparison,
        fall_consented,
        q14_labels,
        spring_augmented,
        spring_consented,
    )


@app.cell(hide_code=True)
def _(augmented, demographics):
    # ── Participants: demographics and scale reliability by cohort ──────
    _vars = ["Gender", "Race/ethnicity", "English main language at home",
             "Prior research experience", "Class year"]
    _rows, _tests = [], []
    for _v in _vars:
        _tab = pd.crosstab(demographics[_v], demographics["cohort"])
        _pct = (100 * _tab / _tab.sum()).round(1)
        for _level in _tab.index:
            _rows.append({"variable": _v, "level": _level,
                          **{f"{_c} n": _tab.loc[_level, _c] for _c in _tab.columns},
                          **{f"{_c} %": _pct.loc[_level, _c] for _c in _tab.columns}})
        _tests.append({"variable": _v, "χ² p (cohort difference)": round(stats.chi2_contingency(_tab)[1], 3)})
    participant_table = pd.DataFrame(_rows)

    def _alpha(items):
        # Cronbach's alpha: k/(k−1) · (1 − Σ item variances / total-score variance)
        _k = items.shape[1]
        return _k / (_k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))

    _reliability = pd.DataFrame([
        {"cohort": _c, "phase": _p,
         "Cronbach's α": round(_alpha(_d.filter(regex=f"^{_p}_")), 3)}
        for _c, _d in augmented.groupby("cohort", sort=False) for _p in ("Pre", "Post")
    ])
    _age = demographics.groupby("cohort", sort=False)["Age"].agg(["count", "mean", "min", "max"]).round(1)

    mo.vstack([
        mo.md("### Participants: demographics by cohort (analysed students)"),
        mo.ui.table(participant_table, selection=None),
        mo.md("**Cohort comparison** (χ² on counts; small cells, read as descriptive)"),
        mo.ui.table(pd.DataFrame(_tests), selection=None),
        mo.md("**Age**"),
        mo.ui.table(_age.reset_index(), selection=None),
        mo.md("**SURE scale reliability** (17 items)"),
        mo.ui.table(_reliability, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Analysis table → `analysis2.xlsx`

    One row per participant, **Q14 self-efficacy** for both cohorts, from
    `Undergrad 2019(1).xlsx` (consent, Q14, demographics; IDs corrected by
    Pascal). Inclusion: consent = 1 and complete Q14 at pre and post. Duplicate
    submissions resolve to the first complete one.

    The graph-interpretation item is analysed separately (Graph task section).
    """)
    return


@app.cell(hide_code=True)
def _():
    # ── ERP graph task: pre-specified two-model LLM grading ─────────────
    # Grading and analysis live in grading/: ERP_ANALYSIS_PLAN.md (written
    # before grading), erp_llm_grade.py (cross-fitted, cohort-blind, consented
    # text only), erp_analysis.py. This cell reads the saved grades and runs the
    # same analysis functions, so the numbers match the script exactly.
    from grading import erp_analysis as _erp

    erp_grades, _raw = _erp.load()
    erp_validation = _erp.validation(erp_grades, _raw)
    erp_results = _erp.analysis(erp_grades)
    _primary = erp_results["effects"].iloc[0]

    mo.vstack([
        mo.md(f"""
    ## ERP graph task

    Students wrote an interpretation of an ERP plot before (Q20) and after (Q22).
    Two graphs were counterbalanced across lab sections: flat (v1, conditions
    identical) and divergent (v2, condition 1 larger from ~350 ms). Each answer
    is graded 0–4 against its own graph's rubric, plus a misconception flag.

    **Graders.** Claude Opus 5.5 and GPT-5.6, 3 passes each; per model the median
    grade and majority flag; the primary grade is the mean of the two models.
    Graders saw the rubric and Morgan's annotated grades as examples, never the
    cohort, time point, or student ID. Spring answers were cross-fitted (5 folds),
    so no answer was graded by a prompt containing its own human grade.

    ### 1. Grader validation (spring answers with human grades)

    Pre-specified bar: two-model mean vs Morgan QWK ≥ 0.64. Result:
    **{erp_validation["mean_qwk"]:.2f} →
    {"confirmatory" if erp_validation["confirmatory"] else "exploratory"}**.
    """),
        mo.ui.table(erp_validation["grades"], selection=None),
        mo.md(f"""
    **Misconception flag** (secondary). Flag rate on human-graded spring answers:
    Morgan {erp_validation["flag_rates"]["Morgan"]}%, Opus
    {erp_validation["flag_rates"]["Opus"]}%, GPT {erp_validation["flag_rates"]["GPT"]}%.
    """),
        mo.ui.table(erp_validation["flags"], selection=None),
        mo.md("**Pass-to-pass consistency** (same model, same answer)"),
        mo.ui.table(erp_validation["passes"], selection=None),
        mo.md(f"""
    ### 2. Cohort comparison

    Students with a gradeable pre and post answer. OLS: post grade ~ cohort + pre
    grade + post graph version, HC3 standard errors.

    **Primary:** spring − fall = {_primary["cohort β (spring − fall)"]:+.3f}
    {_primary["95% CI"]}, p = {_primary["p"]:.3f}. No program effect on ERP
    interpretation.
    """),
        mo.ui.table(erp_results["effects"], selection=None),
        mo.md("**Mean grade (two-model mean) by cohort × graph × time**"),
        mo.ui.table(erp_results["descriptive"], selection=None),
        mo.md("**Secondary: misconception flag at post** (logit: flag ~ cohort + pre flag + post graph)"),
        mo.ui.table(erp_results["flags"], selection=None),
        mo.md("**Sensitivity 3: spring only, Morgan's human grades, pre vs post** (different students per graph; Mann–Whitney)"),
        mo.ui.table(erp_results["spring_human"], selection=None),
    ])
    return erp_grades, erp_results, erp_validation


@app.cell(hide_code=True)
def _(augmented, erp_grades, erp_results, erp_validation):
    # ── Workbook: self-efficacy plus ERP graph-task results ─────────────
    _book = mo.notebook_location() / "raw_data" / "analysis2.xlsx"
    with pd.ExcelWriter(_book, engine="openpyxl") as _writer:
        augmented.to_excel(_writer, sheet_name="self_efficacy", index=False)
        erp_grades.drop(columns="text").to_excel(_writer, sheet_name="erp_grades", index=False)
        erp_validation["grades"].to_excel(_writer, sheet_name="erp_grade_agreement", index=False)
        erp_validation["flags"].to_excel(_writer, sheet_name="erp_flag_agreement", index=False)
        erp_results["effects"].to_excel(_writer, sheet_name="erp_cohort_effect", index=False)
        erp_results["descriptive"].to_excel(_writer, sheet_name="erp_descriptive", index=False)
        erp_results["flags"].to_excel(_writer, sheet_name="erp_flag_model", index=False)
    mo.md(f"Workbook written: `raw_data/{_book.name}` (self-efficacy + ERP sheets).")
    return


if __name__ == "__main__":
    app.run()

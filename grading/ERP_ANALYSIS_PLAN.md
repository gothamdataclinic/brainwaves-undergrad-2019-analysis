# ERP graph task: pre-specified grading and analysis plan

Written 2026-09-25, before any new LLM grading was run. Results produced under
this plan are reported whatever they show, including a null.

## Data

- Only consented students (consent = 1 in `raw_data/Undergrad 2019(1).xlsx`).
  No text from non-consented students is sent to any model, including as a
  grading example.
- Answers: spring pre (Q20) and post (Q22) from the raw Qualtrics exports; fall
  post (Q22) from the raw exports; fall pre (Q20) from
  `PascalFall2019_PrePost.xlsx` (no raw fall pre export exists). Duplicate
  submissions resolve to the first complete one.
- Graph version follows the counterbalancing: sections v1 saw the divergent
  graph (version 2) at pre and the flat graph (version 1) at post; sections v2
  the reverse. Confirmed for spring against the human grading files (100%),
  and for fall from answer content (flat-graph answers say "same/similar").

## Graders

- Two models, Claude Opus 5.5 and GPT-5.6, each run 3 times on every answer.
- Instructions: the two rubrics verbatim, the credited-clause-to-grade rule
  from Morgan's annotated grading, the misconception guidance, and the edge-case
  rules. Grading examples are Morgan's annotated, consented spring answers.
- **Cross-fitting:** spring answers are split into 5 folds; each spring answer
  is graded with examples drawn only from the other folds, so no answer is
  graded by a prompt that contains its own human grade. Fall answers are graded
  with examples from all folds.
- Per model, an answer's grade is the median of its 3 passes; its
  misconception flag is the majority of 3.
- **Primary grade = mean of the two models' per-answer medians.**

## Grader validation (spring, human-graded answers)

Report for each model and for the two-model mean, against Morgan (all
human-graded consented answers) and Rhonda (the shared subset):
exact agreement, agreement within ±1, quadratic-weighted kappa; for the flag,
agreement and Cohen's kappa. Also report agreement between the two models and
across passes within each model.

Human benchmark (consented, n = 27): Morgan–Rhonda exact 70.4%, ±1 96.3%,
QWK 0.74; flag agreement 77.8%, kappa 0.44.

The LLM grade is used for the cohort comparison if its agreement with Morgan
is at least the human benchmark minus 0.10 QWK (i.e. QWK ≥ 0.64). If it falls
short, the cohort comparison is reported as exploratory.

## Primary analysis

- Sample: students with a gradeable pre and post answer.
- Outcome: 0–4 grade at post (primary grade above).
- Model: OLS, post grade ~ cohort + pre grade + post graph version, HC3
  standard errors. The cohort coefficient is the program effect. Two-sided,
  α = .05.

## Secondary analyses

1. Misconception flag at post: logistic regression, post flag ~ cohort + pre
   flag + post graph version.
2. Descriptive: mean grade by cohort × time × graph version.

## Sensitivity analyses

1. Each model separately.
2. Excluding "couldn't see the graph" non-answers (graded 0 with review flag).
3. Spring only: pre-to-post change using Morgan's human grades, split by graph
   version.

"""Cross-fitted LLM grading of the ERP graph task (see ERP_ANALYSIS_PLAN.md).

Usage: .venv/bin/python grading/erp_llm_grade.py MODEL [MODEL ...]
Writes one JSON line per graded batch to grading/erp_llm_raw.jsonl; reruns
skip batches already written, so an interrupted run resumes.
"""

import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
OUT = HERE / "erp_llm_raw.jsonl"
PASSES, FOLDS, BATCH = 3, 5, 12
GATEWAY = os.environ.get("ERP_GATEWAY", "http://127.0.0.1:4000/v1/chat/completions")

RUBRIC = """You are grading undergraduates' written interpretations of an ERP graph.
Task shown to students: "The following graph depicts the results of an EEG
experiment. Describe how you would interpret the graph." The plot showed two
overlaid waveforms (Condition 1 solid, Condition 2 dashed), time in ms, potential in µV.

Two different graphs exist. Grade each answer only against its own version.

VERSION 1 (flat): the two conditions do NOT differ. Up to 4 points:
- [V1-A] Similar responses to condition 1 and condition 2 — 2 points
- [V1-B] Positive peak around ~400 ms — 1 point (1/2 point for incorrect polarity)
- [V1-C] Negative peak around ~100 ms — 1 point (1/2 point for incorrect polarity)
Sample full-credit answer: "Conditions 1 and 2 are nearly identical, and there is a
positive peak at 400 ms and a negative peak of -3 µV at 150 ms."

VERSION 2 (divergent): Condition 1 is stronger. Up to 4 points:
- [V2-A] Condition 1 has a stronger response than condition 2 — 2 points
- [V2-B] This difference emerges around 350 ms — 2 points
Sample full-credit answer: "Condition 1 and 2 elicit the same response until around
300 ms, where Condition 1 elicits a stronger response than Condition 2."

How the reference grader (a trained TA) turned clauses into grades:
- Decide clause by clause first, then map to a grade. Never grade holistically.
- Version 2: 0 clauses -> 0 (or 1 if the answer gestures at the graph's structure
  without a scoreable claim); 1 clause -> 2; both clauses -> 4. Never 3.
- Version 1: 0 clauses -> 0 (or 1 as above); V1-A alone -> 2; two clauses -> 3;
  all three -> 4, but 3 if a peak is described with the wrong polarity.
- Integer grades only; never half points.

Misconception flag (Yes/No), judged separately from the grade:
- Yes for: reading the scalp ERP as single-neuron activity (action potential,
  firing rate, threshold, refractory period, hyperpolarization, resting potential,
  conductance applied to the trace); a reversed or fabricated condition difference;
  a blunt unqualified claim of no difference/effect that misreads the task; axis or
  unit confusion; vacuous non-interpretation.
- No for: hedged similarity claims, admitting inexperience, incomplete but correct
  answers, "graph didn't load" non-answers, loose but defensible wording such as
  "brain activity" for amplitude.
- Grades 3 and 4 are essentially never flagged.

Edge cases: if the answer makes no interpretive claim about the data (e.g. the
graph did not display), grade 0, flag No, flag_review 1. If the student reports a
display problem but still makes a substantive claim, grade the claim normally."""

FORMAT = """Return JSON only, no prose, in exactly this shape:
{"grades": [{"key": "<key>", "credited_clauses": ["V1-A"], "grade": 2,
  "misconception": "No", "misconception_type": "", "flag_review": 0,
  "rationale": "<= 25 words"}]}
misconception_type is one of level_confusion, reversed_condition, unqualified_null,
axis_units, vacuous, or empty. Grade every key given, once."""


def token():
    return subprocess.run(["omp", "auth-gateway", "token"], capture_output=True,
                          text=True, check=True).stdout.strip()


def examples(pool, version):
    rows = pool[pool["graph"].eq(version)]
    return "\n\n".join(
        f"Answer: {r.text}\nCredited clauses: {r.Credited_Clauses or 'none'}\n"
        f"Grade: {r.Morgan_Grade}\nMisconception: {r.Morgan_Misc}"
        for r in rows.itertuples()
    )


def batches(items):
    """(batch_key, version, fold, rows): spring cross-fitted, fall uses all folds."""
    graded = items[items["text"].fillna("").str.strip().ne("")]
    for fold in [*range(FOLDS), None]:
        part = graded[graded["fold"].eq(fold)] if fold is not None else graded[graded["cohort"].eq("Fall")]
        for version in (1, 2):
            rows = part[part["graph"].eq(version)].sort_values("item_id")
            for start in range(0, len(rows), BATCH):
                yield f"f{fold}-v{version}-{start}", version, fold, rows.iloc[start:start + BATCH]


def grade(model, pass_no, key, version, fold, rows, pool, auth):
    exemplars = pool if fold is None else pool[pool["fold"].ne(fold)]
    rng = random.Random(f"{model}|{pass_no}|{key}")
    rows = rows.sample(frac=1, random_state=rng.randrange(2**32))
    # Neutral keys: the grader never sees cohort, time point, or student ID.
    keys = {f"A{i + 1}": item for i, item in enumerate(rows["item_id"])}
    answers = "\n\n".join(f"[{k}] {t}" for k, t in zip(keys, rows["text"]))
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"Graded examples from the reference grader, all VERSION {version}:\n\n"
            f"{examples(exemplars, version)}\n\n---\n\n"
            f"Grade these VERSION {version} answers.\n\n{answers}\n\n{FORMAT}"
        )},
    ]
    for attempt in range(3):
        try:
            r = requests.post(GATEWAY, headers={"Authorization": f"Bearer {auth}"},
                              json={"model": model, "messages": messages}, timeout=900)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            body = content[content.find("{"):content.rfind("}") + 1]
            grades = json.loads(body)["grades"]
            if {g["key"] for g in grades} != set(keys):
                raise ValueError("grader skipped or invented keys")
            for g in grades:
                g["item_id"] = keys.pop(g["key"]) if g["key"] in keys else None
            return {"model": model, "pass": pass_no, "batch": key, "grades": grades}
        except (requests.RequestException, ValueError, KeyError) as error:
            last = error
    return {"model": model, "pass": pass_no, "batch": key, "error": str(last)}


def main(models):
    items = pd.read_csv(HERE / "erp_items.csv", dtype={"Morgan_Grade": str, "Rhonda_Grade": str},
                        keep_default_na=False)
    items["graph"] = items["graph"].astype(int)
    spring = items["cohort"].eq("Spring")
    # Fold by student so both of a student's answers share a fold.
    ids = sorted(items.loc[spring, "ID"].unique())
    fold_of = {i: n % FOLDS for n, i in enumerate(random.Random(0).sample(ids, len(ids)))}
    items["fold"] = items["ID"].map(fold_of).where(spring)
    pool = items[spring & items["Annotated"].eq("1") & items["Morgan_Grade"].isin(list("01234"))]

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            rec = json.loads(line)
            if "grades" in rec:
                done.add((rec["model"], rec["pass"], rec["batch"]))
    jobs = [(m, p, *b) for m in models for p in range(1, PASSES + 1) for b in batches(items)
            if (m, p, b[0]) not in done]
    print(f"{len(jobs)} batches to grade")
    auth = token()
    with ThreadPoolExecutor(6) as pool_exec, OUT.open("a") as out:
        for rec in pool_exec.map(lambda j: grade(*j, pool, auth), jobs):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["model"], rec["pass"], rec["batch"], "ERROR " + rec["error"] if "error" in rec else "ok")


if __name__ == "__main__":
    main(sys.argv[1:])

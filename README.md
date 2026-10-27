# BrainWaves undergraduate study (2019) — analysis code

Analysis code and grading materials for the manuscript *BrainWaves: EEG
research software designed for the classroom* (Aperture Neuro, Neuroscience
Education), reporting a four-week BrainWaves module in an undergraduate
cognitive neuroscience course (NYU PSYCH-UA 25, spring 2019) against a
nonequivalent fall 2019 comparison cohort.

The BrainWaves application itself lives at
https://github.com/makebrainwaves/BrainWaves (archived at
https://doi.org/10.5281/zenodo.23085809). This repository contains only the
study analysis — no application code.

## Contents

| File | What it does |
|---|---|
| `explore.py` | Marimo notebook: builds the analysis table from the master workbook and raw survey exports, and runs the self-efficacy analyses (composite and 17 SURE items; change-score and ANCOVA comparisons, Holm correction) plus the ERP grader-validation and cohort analyses. |
| `make_figures.py` | Renders the manuscript's Table 1, Table 2, and Figure 1 as 300-dpi PNGs from `raw_data/analysis2.xlsx`. |
| `test_explore.py` | Unit tests for the notebook's table loader (synthetic data only). |
| `grading/ERP_ANALYSIS_PLAN.md` | Pre-specified grading and analysis plan for the ERP interpretation task, written before grading was run. |
| `grading/erp_llm_grade.py` | Grades consented students' ERP interpretations with two language models (three passes each), blind to cohort, time point, and student ID, with cross-fitting for spring answers. |
| `grading/erp_analysis.py` | Validates language-model grades against the human graders and runs the pre-specified cohort comparison. |

## Data availability

**No student data are published in this repository.** The study data contain
students' survey responses and free-text answers, including records from
students who did not consent to research use; those records were excluded from
all analyses. Student-level data are therefore withheld, in line with the
study's ethics approval (NYU IRB-FY2017-1171) and consent terms.

What this means in practice:

- `raw_data/` (survey exports, master workbook, grading spreadsheets) is
  listed in `.gitignore` and is not tracked. The code expects it locally.
- Student IDs of excluded, non-consenting records are not in this repository.
  `explore.py` reads them from a local, unpublished file
  (`raw_data/excluded_fall_ids.txt`) when present.
- Grading files that contain student free-text responses
  (`erp_items.csv`, grade files, model output logs) and the calibration brief
  containing verbatim student answers are withheld for the same reason. The
  rubric itself is reproduced in `grading/erp_llm_grade.py`, and the grading
  protocol in `grading/ERP_ANALYSIS_PLAN.md`.
- The de-identified aggregate results underlying the manuscript's tables are
  reproducible from the code given the raw data.

Researchers who wish to discuss access arrangements should contact the
corresponding author of the manuscript.

## Running

Requires Python 3.14 (see `requirements.txt`).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest test_explore.py   # loader tests (no data needed)
.venv/bin/marimo edit explore.py               # full notebook (needs raw_data/)
.venv/bin/python make_figures.py               # tables/figure (needs raw_data/analysis2.xlsx)
```

The ERP grading script additionally requires access to the language-model
gateway it was run against (configured by the `ERP_GATEWAY` environment
variable); it is published for transparency of method, not as a turnkey
pipeline.

## License

MIT (see `LICENSE`). If you use this code, please cite the manuscript and the
BrainWaves software (https://doi.org/10.5281/zenodo.23085809).

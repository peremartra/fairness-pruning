# analysis/

Scripts that turn the experiment data stored on Drive into the tables and figures used in `R1_exp.md` and in the paper.
They only read data that already exists: they never run a model, need no GPU and run on a laptop.

| Folder | Experiment | Status |
|---|---|---|
| [`e2/`](e2/) | E2: random and cross controls, Llama-3.2-3B | first run 2026-11-04 |

## Conventions

- One subfolder per experiment, with its own `README.md` (inputs, how to run, what is written).
- Inputs are small summary files wherever possible. The per-item logs of lm-eval (about 700 MB each) are read only by
  dedicated scripts, once, and their result is stored as a small file.
- Outputs go to a new timestamped folder next to the inputs on Drive (not into the repository), together with an
  `inputs.json` that records every file read (size, sha256) and the versions used.
- The scripts describe; they do not judge. No thresholds and no verdicts: which numbers go to `R1_exp.md` or to the paper is
  decided afterwards.
- Facts that are not stored in the data (for example the GPU that ran each BBQ evaluation) are written in the script as
  constants with a pointer to `R1_exp.md`, and the outputs say so.

Requirements: python >= 3.9, `numpy`, `pandas`; `matplotlib` for figures; `ijson` (optional) to stream large logs.

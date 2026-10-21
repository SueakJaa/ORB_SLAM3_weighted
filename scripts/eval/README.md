# Evaluation pipeline (run -> evaluate -> report)

One config file describes an experiment: which binary, which sequences, which variants
(a variant = a set of `DSLAM_*` environment switches), and how many runs. Three commands do the rest.
Run them from `~/ORB_SLAM3`:

```bash
python3 scripts/eval/run_experiments.py scripts/eval/configs/baseline_tum_dynamic.yaml   # run SLAM (resumable)
python3 scripts/eval/evaluate.py        scripts/eval/configs/baseline_tum_dynamic.yaml   # ATE + RPE per run
python3 scripts/eval/report.py          scripts/eval/configs/baseline_tum_dynamic.yaml   # tables, tests, figures
```

Results go to `results/<experiment name>/`:

```
<sequence>/<variant>/run_01/   traj.txt  kf.txt  log.txt  meta.json (settings, git commit, timing)
metrics.csv                    one row per run: ATE, RPE, keyframes, tracking time, crash status
report/summary.md              tables per sequence + Mann-Whitney U test against the baseline variant
report/ate_box.png, rpe_box.png, traj_<seq>.png, error_<seq>.png
```

## Configs

| File | What it runs |
|---|---|
| `configs/baseline_tum_dynamic.yaml` | plain ORB-SLAM3, 7 fr3 dynamic sequences (walking + sitting), 10 runs |
| `configs/thesis_main.yaml` | A0 (ORB-SLAM3) vs SW_G10 (YOLO-only switch) vs GEOK2_G10 (final method), 12 sequences |

New experiment: copy a config, change `name`, `variants`, `sequences`. Never reuse a `name` for different
settings - the runner skips runs that already exist under that name.

## Metrics (checked against evo 1.38 on synthetic data: identical values)

- **ATE**: positions associated by timestamp (|dt| <= 0.01 s), SE(3) Umeyama alignment (no scale), RMSE
  of the position error. Same number as `evo_ape tum gt.txt est.txt -a`.
- **RPE**: relative motion error over `rpe_delta` (default 1 s, the TUM benchmark convention = drift per
  second), translation in m and rotation in degrees, all overlapping pairs. With `rpe_delta_unit: f` and
  `rpe_delta: 1` it equals `evo_rpe tum gt est` (and `--all_pairs` for larger frame deltas).
- **Failure**: ATE > `fail_threshold_m` (default 0.1 m) or no trajectory (crash). Crashes are kept as
  results; `--retry-crashed` re-runs only those, `--force` re-runs everything.
- **Significance**: two-sided Mann-Whitney U test of ATE, each variant vs the `baseline` variant.

## Rules that keep results comparable

- Runs are sequential on purpose. Do not run two experiments at the same time: ORB-SLAM3's keyframe
  decisions depend on processing speed.
- The runner warns when the code has uncommitted changes; `meta.json` records the git commit and the
  exact `DSLAM_*` settings of every run - check them before comparing numbers.
- Rebuild (`cd build && make -j4`) after switching code versions.

## Older results

`import_legacy.py` copies results of the old shell scripts (`results_v16/`, `results_v18_unseen/`, ...)
into this layout so they can be evaluated and plotted without re-running:

```bash
python3 scripts/eval/import_legacy.py results_v16 scripts/eval/configs/thesis_main.yaml
```

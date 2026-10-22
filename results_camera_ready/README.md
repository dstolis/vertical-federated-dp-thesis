# Camera-ready results (AICS 2026)

Per-round logs for every run reported in the camera-ready paper. One CSV per run;
the last row (`noise_lock == -1`) is the restored best-validation model,
re-evaluated on the validation and test splits.

Protocol: 10,000 samples held out from the official training split (fixed seed
2026) for checkpoint selection and the noise-lock; the official test split is
only used to report the selected model.

Reproduce:

    python scripts/run_camera_ready.py list          # all 72 jobs
    python scripts/run_camera_ready.py run           # run (several workers can share the queue)
    python scripts/collect_ir_norms.py               # ir_norms/*.npy for Fig. 2a
    python scripts/summarise_camera_ready.py <paper_dir>   # tables + numbers.tex
    python scripts/verify_camera_ready.py <paper_dir>      # independent re-check

Tags: `s{seed}_{dataset}_{mode}_z{z}` (mode: baseline, dp, adaptive,
dpC5 = fixed C=5, dpC5lr = fixed C=5 with lr/5, dpC5m = fixed C=5 with the
adaptive configuration's data order), `noln_*` (no LayerNorm), `v6_*`/`v7_*`
(Flower-style server-side adaptive clipping, with / without our tuner).

Cumulative epsilon: one release per local epoch per party; k releases compose
to a Gaussian mechanism with multiplier z/sqrt(k), solved exactly with the
analytic Gaussian mechanism (Balle and Wang, 2018) at delta = 1e-5.

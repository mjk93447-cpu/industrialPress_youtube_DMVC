# Executable PoC Development Plan

## What The Current CI Artifact Is

The GitHub Actions workflow `build-artifact` produces `dmvc-poc-spec-bundle.zip`, a **documentation and contract bundle** only. It contains:

- Markdown specifications
- JSON Schema and static configuration files
- A small JSON syntax validator

It does **not** install a training environment, run model code, or process video. It exists to keep specifications versioned and machine-checkable. This is separate from a **runnable** PoC that can train, evaluate, and run inference.

## Definition Of Success For A Runnable PoC

An executable PoC must ship software that a developer can run locally or in CI with documented commands and produce measurable outputs:

| Capability | Minimum acceptance |
| --- | --- |
| Data ingest | Import approved clips into a manifest-backed dataset index; optional YouTube **metadata-only** candidate crawl using API keys from environment |
| Preprocessing | Extract frames and mono audio at fixed FPS/sample rate; compute optical flow or frame-diff features; emit tensors or LMDB/WebDataset shards |
| Training | Single-GPU script trains baseline models from `docs/03_model_architecture.md` (RGB → RGB+motion+audio → DMVC-lite) with logging checkpoints |
| Evaluation | Script computes metrics from `docs/04_evaluation_plan.md` on validation/test splits; exports CSV/JSON |
| Inference demo | CLI or notebook: given a **pre-failure** clip segment, outputs precursor timeline + 2D damage heatmap (probabilistic) |
| Reproducibility | `requirements.txt` or `pyproject.toml` pins versions; one-command smoke test on **synthetic or tiny public fixtures** |

Full 3D volumetric fracture prediction remains **out of scope** for the first executable PoC unless multi-view or synthetic ground truth is added later.

## Recommended Technology Stack

These choices prioritize a small team and Windows/Linux parity:

- **Language:** Python 3.11+
- **Deep learning:** PyTorch 2.x
- **Video I/O:** PyAV or ffmpeg-python; **audio:** torchaudio or librosa
- **Flow:** torchvision `optical_flow` helpers or RAFT small variant if budget allows
- **Training orchestration:** Lightning optional; plain scripts acceptable for PoC
- **Config:** Hydra or YAML + argparse
- **Experiment tracking:** TensorBoard minimum; Weights & Biases optional
- **Packaging:** Editable install `pip install -e .` from repository root

## Repository Layout (Target)

Introduce gradually; not all directories exist on day one.

```
src/dmvc_poc/           # Library code
  data/                 # datasets, manifests, collate
  models/               # encoders, fusion, heads
  train/                # train_loop, losses
  eval/                 # metrics, reports
  infer/                # CLI demo
scripts/                # youtube_metadata_crawl.py, preprocess.py, train.py, evaluate.py, demo_infer.py
tests/                  # unit + smoke tests on synthetic tensors
fixtures/synthetic/     # tiny tensors or 2-frame clips committed for CI
configs/train/          # Hydra/YAML overrides
```

## Phased Development Plan

### Phase 0 — Foundation (1–2 weeks)

- Add `pyproject.toml` or `requirements.txt` with pinned ML dependencies.
- Wrap existing JSON Schema validation in `pytest` or keep `scripts/validate_json.py` and call it from CI.
- Create **synthetic fixtures**: random tensors simulating `(B,T,C,H,W)` video and `(B,F)` mel specs to allow **training smoke tests without real video**.
- Extend CI: job `smoke-train` runs 1–2 optimizer steps on CPU or small GPU runner (optional `workflow_dispatch` only if GPU cost is an issue).

**Exit:** `pytest` passes; smoke train completes in under 5 minutes on CPU.

### Phase 1 — Data Pipeline (2–4 weeks)

- Implement `Dataset` reading manifests validated against `schemas/dmvc_clip.schema.json`.
- Load frame folders + WAV segments; apply temporal cropping ending **before** `t_break` for training inputs.
- Optional: `scripts/youtube_metadata_crawl.py` using YouTube Data API (metadata only; secrets via env); outputs Parquet/CSV into `data/candidates/` (gitignored).
- Document clip ingestion for **rights-approved** assets only (see `docs/06_youtube_data_pipeline.md`).

**Exit:** One command builds a preprocessed cache for N approved clips; unit tests use synthetic or 1–2 committed CC0 micro-clips if legally cleared.

### Phase 2 — Models And Training (3–5 weeks)

Implement the baseline ladder in code:

1. `rgb_temporal_baseline` — 2D CNN backbone + temporal pooling.
2. Add motion branch (flow stacks).
3. Add audio branch (log-mel CNN).
4. `dmvc_lite` — view dropout + simple gating MLP (full transformer fusion optional stretch goal).

Joint heads:

- Precursor / `break_within_K` classification.
- Optional soft Dice loss on 2D damage heatmap vs masks when available.

**Exit:** Training loss decreases on validation; checkpoints saved; TensorBoard logs exist.

### Phase 3 — Evaluation And Reporting (1–2 weeks)

- Implement metric functions: AUROC, soft-IoU, median timing error.
- `scripts/evaluate.py` writes `reports/metrics.json` and CSV slices by material/geometry.
- Bootstrap confidence intervals for small-N datasets.

**Exit:** Single command reproduces the evaluation table for a given checkpoint.

### Phase 4 — Inference Demo And Packaging (1–2 weeks)

- `scripts/demo_infer.py --clip path_or_id --end_sec T` outputs PNG heatmap + JSON timeline.
- Optional Gradio app behind feature flag.
- Release workflow: build **wheel** or **Docker image** artifact (distinct from the spec zip), tagged `v0.x-poc`.

**Exit:** Non-ML stakeholder can run demo on a held-out clip with written instructions.

## CI Artifact Evolution

| Stage | Artifact | Purpose |
| --- | --- | --- |
| Now | `dmvc-poc-spec-bundle` | Specs and schemas |
| Phase 0+ | `dmvc-poc-wheel` or `docker-image` | Installable package |
| Phase 2+ | `checkpoint-smoke` (optional) | Tiny checkpoint from smoke train (not for production) |

Keep the spec bundle job if specifications remain useful for auditors; add **parallel** jobs for real software artifacts.

## Dependencies And Risks

- **GPU access:** Local GPU or cloud budget required for meaningful training; CPU-only smoke tests mitigate CI cost.
- **Data rights:** Executable pipeline must enforce rights gate before storing pixels; metadata crawl remains separate.
- **Label noise:** Invest in reviewer tooling (semi-automatic `t_break` proposals) early to avoid garbage-in.
- **Scope creep:** Defer full DMVC transformer fusion until tri-modal baseline beats RGB-only.

## Milestone Table (Indicative)

| Week range | Milestone |
| --- | --- |
| 1–2 | Package layout, synthetic smoke train, CI smoke |
| 3–6 | Approved-clip preprocessing + baseline RGB trainer |
| 7–10 | Motion + audio + DMVC-lite + evaluation scripts |
| 11–12 | Inference demo + documentation + v0.1 PoC tag |

Adjust durations after Phase 0 depending on team size and GPU availability.

## Immediate Next Actions

1. Approve Python packaging layout and dependency pin strategy.
2. Decide whether CI smoke train runs on every push or only `workflow_dispatch` / `main`.
3. Allocate storage for approved clips outside git (network drive or object storage).
4. Prioritize **one** baseline (RGB+motion) if timeline slips—heatmap head can trail precursor head by one sprint.

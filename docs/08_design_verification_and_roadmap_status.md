# Design Verification And Roadmap Status

## Purpose

This document ties together:

- Written specifications (`docs/01`–`docs/07`)
- Executable checks (`pytest`, CPU smoke training)
- A quantitative **roadmap completion ratio** for required deliverables tracked in `src/dmvc_poc/roadmap/status.py`

It answers whether implementation stays aligned with design *right now*, and how complete or stable the codebase is relative to the original PoC roadmap.

## How Verification Works

| Layer | What it proves |
| --- | --- |
| JSON syntax | `scripts/validate_json.py` (CI) |
| JSON Schema contracts | `tests/test_manifest_schema.py` validates `fixtures/synthetic/sample_clip_manifest.json` against `schemas/dmvc_clip.schema.json` |
| Config alignment | `tests/test_config_alignment.py` asserts critical keys in `configs/*.json` match documented evaluation constraints |
| Runnable ML core | `tests/test_smoke_training.py` runs one Adam step on synthetic video tensors through `RGBTemporalBaseline` |
| Metrics sanity | `tests/test_metrics.py` covers soft IoU and AUROC helpers used later for evaluation |
| Roadmap compliance | `tests/test_roadmap_compliance.py` ensures Phase 0 paths exist and optional Phase 2–4 placeholders remain declared |

Stability baseline: CI runs `pytest` on every push; all tests must pass for merges to stay green.

## Quantitative Status

Run locally:

```bash
pip install -e ".[dev]"
pytest
python scripts/roadmap_report.py
```

Interpreting output:

- **Required completion ratio** counts only non-optional checklist rows whose paths exist.
- Optional rows (`scripts/train.py`, `scripts/evaluate.py`, etc.) remain documented placeholders until later phases.

**Snapshot (generated via `python scripts/roadmap_report.py` on a clean checkout):** required completion **100%** (Phase 0 + minimal Phase 1 manifest index). Optional CLIs for preprocessing, training, evaluation, and inference still read as **not present** in the repository, which is expected until Phases 1–4 land.

## Qualitative Assessment (Living Document)

Update this section whenever major milestones land.

### Alignment With Original PoC Documents

| Document topic | Implementation coverage |
| --- | --- |
| Rights-gated acquisition (`docs/01`, `docs/06`) | Operational docs only; **no automated YouTube downloader** by design |
| Dataset schema (`docs/02`) | `ClipManifestIndex` validates manifests; **no frame/audio decoding yet** |
| Model ladder (`docs/03`) | **RGB temporal baseline + heatmap head stub** implemented; motion/audio/DMVC-lite **not implemented** |
| Evaluation (`docs/04`) | Metric helpers started; full reporting CLI **not implemented** |
| Executable plan (`docs/07`) | Phase 0 largely satisfied (packaging, smoke train, tests); Phases 1–4 partially started |

### Completeness Versus Original 12-Week Roadmap (`docs/05`)

Rough staging:

- **Weeks 1–2 (policy + metadata):** Documentation complete; ingestion automation **partial** (schema + manifest index only).
- **Weeks 3–4 (annotation + preprocessing):** Manifest fixture only; preprocessing pipeline **missing**.
- **Weeks 5–7 (baselines):** Minimal RGB baseline code present; full training pipeline **missing**.
- **Weeks 8–12 (DMVC + demo):** Not started in code.

### Stability And Risk Notes

- **Deterministic smoke tests** keep CPU training reproducible with fixed seeds inside `run_smoke_training_step`.
- **Heavy dependencies** (`torch`) increase CI time; acceptable for research PoC but monitor GitHub Actions minutes.
- **Editable installs** expect repository layout with `schemas/` at repo root; non-editable packaging would need packaging rule updates.

## Next Steps To Raise Completion Score

1. Implement real frame/audio loaders + caching (`Phase 1`).
2. Add `scripts/train.py` with configurable horizons and logging (`Phase 2`).
3. Expand tests with saved tiny tensors simulating optical flow and mel features.

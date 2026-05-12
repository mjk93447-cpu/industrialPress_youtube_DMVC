# DMVC Hydraulic Press Failure Prediction PoC

[![CI](https://github.com/mjk93447-cpu/industrialPress_youtube_DMVC/actions/workflows/ci.yml/badge.svg)](https://github.com/mjk93447-cpu/industrialPress_youtube_DMVC/actions/workflows/ci.yml)

This workspace contains the execution artifacts for a 1-3 month PoC that uses approved hydraulic-press video clips to study failure precursor detection and post-failure damage heatmap prediction.

**Remote:** https://github.com/mjk93447-cpu/industrialPress_youtube_DMVC

The PoC scope is intentionally rights-gated and evaluation-driven:

- YouTube automation is used for candidate metadata discovery, not unrestricted audiovisual download.
- Training clips require Creative Commons status, creator permission, owned capture, a separate license, or synthetic origin.
- The first model target is precursor timing plus 2D/2.5D damage heatmaps, not an overclaimed full 3D fracture simulator.
- DMVC value is measured through RGB, motion, audio, and object-context ablations.

## Documents

- `docs/01_data_acquisition_policy.md`: YouTube candidate discovery, rights states, exclusion criteria, and approved training-data paths.
- `docs/02_dataset_schema_and_labeling.md`: Clip schema, event labels, damage masks, quality flags, and split policy.
- `docs/03_model_architecture.md`: Baseline ladder, DMVC fusion, view gating, diversity/alignment losses, and output heads.
- `docs/04_evaluation_plan.md`: Timing, precursor, heatmap, ablation, and generalization metrics.
- `docs/05_timeline_and_risks.md`: 12-week execution plan, risk register, staffing, and decision gates.
- `docs/06_youtube_data_pipeline.md`: Operational guide for YouTube API ingestion, directory layout, rights gating, secrets, and retention.
- `docs/07_executable_poc_development_plan.md`: Plan for a **runnable** PoC (training, evaluation, inference), distinct from the specification-only CI artifact.
- `docs/08_design_verification_and_roadmap_status.md`: Design verification tests, roadmap completion metrics, stability notes.

## Machine-Readable Contracts

- `schemas/dmvc_clip.schema.json`: JSON Schema for approved clip records.
- `configs/dmvc_poc_model.json`: Initial model and ablation configuration.
- `configs/evaluation_metrics.json`: Strict evaluation controls, metric lists, and PoC acceptance targets.

## Continuous Integration

On each push or pull request to `main`, GitHub Actions:

1. Validates JSON syntax for `schemas/` and `configs/`.
2. Installs the Python package in editable mode, runs `pytest`, and prints `scripts/roadmap_report.py`.
3. Uploads `dmvc-poc-spec-bundle.zip` with documentation, JSON contracts, `src/`, `tests/`, `fixtures/`, and helper scripts.

That artifact is a **specification bundle**, not an installable ML package or trained model. See `docs/07_executable_poc_development_plan.md` for the roadmap to trainable software and demo artifacts.

## Development

```bash
pip install -e ".[dev]"
pytest
python scripts/roadmap_report.py
```

## PoC Completion Target

The PoC is validated when it processes at least 50 approved clips, labels at least 20 damage masks, shows measurable value beyond RGB-only baselines, and produces understandable held-out damage heatmaps for expert review.

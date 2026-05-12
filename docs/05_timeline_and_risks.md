# PoC Timeline And Risk Register

## Purpose

This document turns the DMVC hydraulic-press roadmap into a 12-week execution plan with checkpoints, owners, dependencies, and risk controls.

## 12-Week Timeline

### Weeks 1-2: Data Scope And Rights Gate

Goals:

- Implement the acquisition policy from `01_data_acquisition_policy.md`.
- Collect 200-500 candidate metadata records.
- Identify at least 50 likely approvable clips.

Deliverables:

- Candidate metadata table.
- Rights status queue.
- Creator permission template.
- Exclusion log with reasons.

Exit criteria:

- Every candidate has `review_status`.
- Every training candidate has an approved `rights_status`.
- No standard-license YouTube audiovisual content is stored as training data without separate permission.

### Weeks 3-4: Dataset And Annotation

Goals:

- Apply the schema from `02_dataset_schema_and_labeling.md`.
- Produce synchronized frame, audio, event, and mask assets.
- Label at least 50 approved clips and 20 damage masks.

Deliverables:

- Validated clip records matching `schemas/dmvc_clip.schema.json`.
- Event labels for `t_contact`, `t_pre_start`, `t_break`, and `t_after`.
- Damage masks and quality flags.
- Inter-annotator agreement report for 20% of clips.

Exit criteria:

- JSON records validate against the schema.
- Strict test split is group-disjoint.
- Low-quality clips are flagged before training.

### Weeks 5-7: Baseline Models

Goals:

- Train the baseline ladder through `tri_modal_baseline`.
- Establish timing and heatmap baselines before DMVC losses are introduced.

Deliverables:

- `rgb_temporal_baseline` metrics.
- `rgb_motion_baseline` metrics.
- `rgb_audio_baseline` metrics.
- `tri_modal_baseline` metrics.
- First qualitative heatmap cards.

Exit criteria:

- RGB-only timing baseline is reproducible.
- At least one non-RGB view is measured through ablation.
- Center-prior and object-mask-uniform heatmap baselines are reported.

### Weeks 8-10: DMVC PoC Model

Goals:

- Add view dropout, view gating, temporal alignment, diversity penalty, and sparse core-view selection.
- Compare `dmvc_poc` against fixed-fusion and no-loss ablations.

Deliverables:

- DMVC training run summary.
- View importance analysis by material, geometry, and audio quality.
- Alignment/diversity loss sweep report.
- Error analysis for false alarms and missed failures.

Exit criteria:

- `dmvc_poc` beats at least one fixed-fusion baseline metric.
- View-gating behavior is inspectable.
- No evaluation leakage is found.

### Weeks 11-12: Demo And Final Evaluation

Goals:

- Run the demo protocol on held-out clips.
- Prepare a PoC decision report.

Deliverables:

- Inference demo for pre-failure-only input.
- Timeline visualization of precursor probabilities.
- 2D or 2.5D damage heatmap visualization.
- Final metric table with confidence intervals.
- Next-phase recommendation.

Exit criteria:

- 10-30 held-out demo clips have result cards.
- Evaluation metrics follow `04_evaluation_plan.md`.
- Risks and limitations are documented without overclaiming 3D capability.

## Risk Register

| Risk | Probability | Impact | Mitigation | Trigger |
| --- | --- | --- | --- | --- |
| Insufficient approved YouTube clips | High | High | Add owned captures, synthetic clips, and direct creator outreach | Fewer than 50 approved clips by week 3 |
| YouTube policy violation risk | Medium | High | Keep automatic collection to metadata and rights-gated assets only | Any raw YouTube clip without approved rights |
| Labels are too noisy | Medium | High | Add second-review pass and adjudication thresholds | `t_break` disagreement above 0.5 seconds |
| Single-view videos cannot support 3D claims | High | Medium | Treat 3D as 2.5D proxy unless multi-view or synthetic data exists | 3D output requested for monocular-only clips |
| Audio dominated by music or voiceover | High | Medium | Use audio confidence gating and audio ablation | More than 40% clips flagged `background_music` |
| Slow-motion editing distorts timing | Medium | Medium | Flag timewarp clips and evaluate separately | Many candidates have `slow_motion_or_timewarp` |
| DMVC overfits small data | Medium | High | Keep baseline ladder, view dropout, small encoders, and confidence intervals | Validation improves while strict test degrades |
| Heatmap labels too sparse | Medium | Medium | Use sparse point metrics plus mask metrics | Fewer than 20 masks by week 4 |
| Compute budget limits video transformers | Medium | Medium | Start with frame encoder plus temporal pooling | Training runs exceed available GPU budget |
| Model uses text leakage | Low | High | Disable outcome-revealing text in strict evaluation | Metadata mentions exact fracture outcome |

## Staffing Assumption

Minimum PoC team:

- 1 data engineer for API metadata, storage layout, validation scripts, and preprocessing.
- 1 ML engineer for baselines, DMVC model, training, and evaluation.
- 1 annotator or research assistant for labels and quality review.
- 1 domain reviewer for failure-mode sanity checks and qualitative heatmap review.

## Weekly Operating Rhythm

- Monday: Review data count, labeling throughput, and blockers.
- Wednesday: Review model/evaluation runs and error cases.
- Friday: Freeze weekly artifacts and record metric deltas.

Every week should produce a short status note with:

- Approved clip count.
- Labeled clip count.
- Masked clip count.
- Best validation metrics.
- Strict test metrics if available.
- New risks or mitigation changes.

## Next-Phase Decision

After week 12, choose one path:

- Continue with YouTube/creator-permission dataset expansion if precursor timing is promising.
- Move to controlled multi-view capture if 3D heatmaps are the priority.
- Add simulation and synthetic pretraining if rare material/failure combinations block generalization.
- Stop or redesign if RGB-only cannot beat trivial timing baselines or labels are not reliable enough.

# Evaluation Plan

## Purpose

This document defines how the PoC will test precursor detection, failure onset timing, damage heatmap prediction, multimodal value, and generalization to unseen videos.

## Evaluation Splits

Use the group-aware dataset split from `02_dataset_schema_and_labeling.md`.

Strict test split:

- No shared `split_group` with training data.
- No post-failure input frames.
- No failure-mode label as an inference input.
- No text metadata that directly reveals the outcome.
- Exclude clips with `hard_cut_near_failure`, `object_occluded`, or `unclear_rights`.

Stress-test splits:

- `unseen_material`
- `unseen_geometry`
- `unseen_channel_or_session`
- `audio_degraded`
- `low_resolution`
- `slow_motion_or_timewarp`

## Timing Metrics

### Failure Onset Error

Measure the difference between predicted break time and annotated `t_break`.

- `mae_seconds`: Mean absolute error.
- `median_absolute_error_seconds`: Robust headline metric.
- `tolerance_accuracy_0_5s`: Fraction of predictions within 0.5 seconds.
- `tolerance_accuracy_1_0s`: Fraction of predictions within 1.0 seconds.

PoC target:

- Median absolute error at or below 1.0 second.
- 0.5-second tolerance accuracy reported for transparent difficulty tracking.

### Early Warning Lead Time

For each true positive warning, compute:

`lead_time = t_break - t_warning`

Report:

- Median lead time.
- 10th percentile lead time.
- False alarm rate per clip.
- Missed failure rate.

Warnings after `t_break` count as late detections and should be reported separately.

## Precursor Classification Metrics

Evaluate `P(pre_failure)` and `P(break_within_K)` for `K = 0.5s`, `1.0s`, and `2.0s`.

Metrics:

- AUROC.
- AUPRC.
- F1 at selected operating threshold.
- Precision at 80% recall.
- Expected calibration error.

PoC target:

- AUROC at or above 0.80 on validation.
- Test-set F1 at or above 0.75 if enough labeled clips exist.

If the dataset is too small for stable F1, report confidence intervals through bootstrap resampling instead of overclaiming.

## Damage Heatmap Metrics

Evaluate predicted `damage_heatmap_2d` against the post-failure mask.

Metrics:

- `soft_iou`: Intersection-over-union using soft probabilities.
- `dice`: Dice coefficient after thresholding.
- `top_k_localization`: Whether the top-k predicted pixels overlap the damage mask.
- `pointing_game_accuracy`: Whether the maximum probability point falls inside the damage mask.
- `calibration_error`: Whether heatmap probabilities correspond to empirical damage frequency.

PoC target:

- Beat a center-prior baseline and an object-mask-uniform baseline.
- Produce interpretable qualitative heatmaps on 10-30 held-out clips.

## Baseline And Ablation Matrix

Run the same split across all rows.

| Experiment | Views | Purpose |
| --- | --- | --- |
| `center_prior_heatmap` | None | Spatial prior baseline |
| `rgb_temporal_baseline` | RGB | Minimum video baseline |
| `rgb_motion_baseline` | RGB + motion | Test deformation signal value |
| `rgb_audio_baseline` | RGB + audio | Test acoustic precursor value |
| `tri_modal_baseline` | RGB + motion + audio | Fixed multimodal fusion baseline |
| `dmvc_no_diversity` | RGB + motion + audio + context | Alignment without diversity penalty |
| `dmvc_no_alignment` | RGB + motion + audio + context | Diversity without temporal alignment |
| `dmvc_poc` | RGB + motion + audio + context | Full PoC model |

Report both average performance and per-material/per-geometry breakdowns.

## Generalization Analysis

For each model, report metrics by:

- Material category.
- Geometry category.
- Failure mode.
- Channel or capture setup.
- Audio confidence bucket.
- Quality flag bucket.

Required error analysis:

- Top 10 false alarms.
- Top 10 missed failures.
- Top 10 worst heatmap predictions.
- Cases where DMVC gates ignored audio.
- Cases where audio dominated the prediction.

## Statistical Reporting

Because the PoC dataset is small, every headline metric should include:

- Number of evaluated clips.
- Mean and median where appropriate.
- Bootstrap 95% confidence interval for AUROC, F1, and soft-IoU.
- Per-split metric table.

Avoid claiming model superiority unless the same split, labels, and cutoff rule were used.

## Demo Evaluation Protocol

For each unseen demo clip:

1. Hide all frames at and after the inference cutoff.
2. Run the model on the pre-failure clip only.
3. Show precursor probability over time.
4. Show predicted time-to-break distribution.
5. Show predicted damage heatmap.
6. Reveal the actual post-failure frame and mask.
7. Save a side-by-side result card for qualitative review.

## Acceptance Gate

The PoC is considered validated if:

- At least 50 approved clips are processed.
- At least 20 clips have damage masks.
- At least one multimodal model beats RGB-only timing metrics.
- The full DMVC model beats the best fixed-fusion baseline on at least one primary metric without materially degrading the others.
- Held-out qualitative results are understandable enough for a domain expert to judge failure region plausibility.

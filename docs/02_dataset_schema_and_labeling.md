# Dataset Schema And Labeling Guide

## Purpose

This guide defines the PoC dataset contract for hydraulic-press failure prediction. It covers approved clip records, event labels, damage annotations, quality flags, and split rules.

## Dataset Unit

The dataset unit is an approved `clip`, not a full YouTube video. A clip contains the interval from stable pre-contact context through post-failure evidence.

Recommended clip window:

- Start: 2-5 seconds before press-object contact.
- End: 1-3 seconds after the first visible failure or after the main deformation result is visible.
- Frame rate: preserve native frame rate for raw assets; export model-ready tensors at 16, 24, or 30 fps.
- Audio: preserve mono 16 kHz or 22.05 kHz waveform for feature extraction.

## Required Record Fields

Each clip record must include:

- `clip_id`: Stable project-local identifier.
- `source_video_id`: YouTube video ID, internal capture ID, or synthetic ID.
- `source_type`: `youtube`, `owned_capture`, `licensed_source`, or `synthetic`.
- `rights_status`: One approved status from `01_data_acquisition_policy.md`.
- `object_material`: Primary visible material.
- `object_geometry`: Dominant shape category.
- `failure_mode`: Primary observed failure mode.
- `camera_view`: Camera view class and quality.
- `frame_rate`: Raw frame rate if known.
- `duration_seconds`: Clip duration.
- `event_labels`: Temporal labels.
- `damage_labels`: Spatial labels.
- `quality_flags`: Known issues and confidence.
- `split_group`: Group used for leakage-safe splitting, normally channel or capture session.

## Controlled Vocabularies

`object_material`:

- `glass`
- `ceramic`
- `metal`
- `plastic`
- `rubber`
- `wood`
- `food`
- `composite`
- `unknown`

`object_geometry`:

- `sphere`
- `cylinder`
- `box`
- `thin_plate`
- `shell`
- `irregular`
- `unknown`

`failure_mode`:

- `crack`
- `shatter`
- `crumple`
- `burst`
- `delamination`
- `localized_puncture`
- `extrusion`
- `no_clear_failure`
- `unknown`

`camera_view`:

- `front`
- `side`
- `top_oblique`
- `close_up`
- `multi_shot_edited`
- `unknown`

## Temporal Labels

The core temporal labels are:

- `t_contact`: First frame where the press visibly contacts or starts loading the object.
- `t_pre_start`: First frame where pre-failure evidence is visible or audible, such as micro-cracking, sudden vibration change, local buckling, sound onset, or deformation acceleration.
- `t_break`: First frame where irreversible visible damage begins.
- `t_after`: Representative frame after the main failure result is visible.

If a label cannot be determined, set the value to `null` and record a confidence below `0.5`.

Temporal confidence scale:

- `1.0`: Frame-accurate with clear evidence.
- `0.75`: Within a short frame range.
- `0.5`: Approximate but usable.
- `0.25`: Weak evidence; exclude from strict evaluation.

## Damage Labels

The first PoC uses 2D damage labels.

Required damage annotation:

- `post_failure_mask_uri`: Binary or soft mask for the damaged region on `t_after`.
- `object_visible_mask_uri`: Mask of the visible object before failure if available.
- `damage_points`: Optional sparse points marking likely failure origin, crack tips, puncture centers, or shatter zones.
- `damage_confidence`: Annotator confidence from 0 to 1.

Optional 2.5D fields:

- `depth_source`: `none`, `monocular_estimate`, `manual_proxy`, or `synthetic_ground_truth`.
- `canonical_mesh_id`: Identifier for a proxy geometry if the 2D mask is projected onto a canonical object.
- `projected_heatmap_uri`: Storage URI for a canonical 2.5D/3D heatmap.

## Quality Flags

Use zero or more flags:

- `heavy_motion_blur`
- `slow_motion_or_timewarp`
- `hard_cut_near_failure`
- `background_music`
- `dominant_voiceover`
- `object_occluded`
- `low_resolution`
- `non_stationary_camera`
- `multiple_objects`
- `unclear_rights`

Clips with `hard_cut_near_failure`, `object_occluded`, or `unclear_rights` are excluded from strict model evaluation.

## Dataset Splits

The primary split is group-aware:

- Train: 70%
- Validation: 15%
- Test: 15%

Group by `split_group` so clips from the same YouTube channel, creator, owned capture session, or synthetic generation batch do not leak across train/validation/test.

Maintain secondary stress-test splits:

- Unseen material.
- Unseen geometry.
- Unseen channel or capture setup.
- Audio degraded or audio unavailable.

## Labeling Workflow

1. Reviewer confirms the clip is approved for training.
2. Reviewer assigns material, geometry, camera view, and failure mode.
3. Reviewer marks `t_contact`, `t_pre_start`, `t_break`, and `t_after`.
4. Reviewer draws post-failure damage mask on `t_after`.
5. Reviewer adds optional origin points and object mask.
6. A second reviewer relabels at least 20% of clips.
7. Disagreements above 0.5 seconds for `t_break` or below 0.6 mask IoU require adjudication.

## Acceptance Criteria For PoC Dataset

The first usable PoC dataset must contain:

- At least 50 approved clips.
- At least 20 clips with post-failure damage masks.
- At least 4 material categories.
- At least 3 geometry categories.
- At least 10 clips reserved for channel- or session-disjoint testing.

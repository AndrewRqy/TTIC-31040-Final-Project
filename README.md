# Robustness of Relative Camera Pose Estimation Under Image Degradation
**Andrew Ren — TTIC 31040**

Evaluates how common image degradations affect SIFT-based relative pose estimation, and whether classical preprocessing can recover accuracy.

---

## Structure

```
data/
  tum_sequence/     place downloaded TUM sequence here
results/            auto-generated figures and result JSON
main.ipynb          experiment driver
```

**`src/dataset.py`** — TUM RGB-D loader with timestamp association and ground-truth pose extraction
- `_quat_to_rot`, `_pose_to_T` — quaternion/translation → 4×4 transform
- `_associate` — nearest-neighbor timestamp matching between RGB and ground-truth streams
- `TUMDataset` — dataset class: `_load`, `_load_img`, `get_relative_pose`, `get_pairs`, `get_K`

**`src/degradation.py`** — four image corruption types at 5 severity levels each
- `apply_gaussian_noise`, `apply_motion_blur`, `apply_jpeg_compression`, `apply_illumination_shift`
- `apply_degradation` — unified entry point
- `get_severity_label` — human-readable parameter string for a given (type, severity)

**`src/preprocessing.py`** — classical denoising and contrast enhancement methods
- `histogram_equalization`, `clahe`, `gaussian_smoothing`, `bilateral_filter`, `non_local_means`, `intensity_normalization`, `no_preprocessing`
- `apply_preprocessing` — unified entry point

**`src/pipeline.py`** — SIFT feature matching and RANSAC-based relative pose estimation
- `detect_and_match_sift` — keypoint detection + Lowe ratio test matching
- `_pose_from_E`, `_pose_from_H` — pose recovery from essential matrix / homography
- `run_essential_pipeline`, `run_homography_pipeline` — full pipelines returning `(R, t, inlier_mask, ...)`
- `run_pipeline` — unified entry point with optional preprocessing

**`src/metrics.py`** — pose evaluation metrics (all in degrees or pixels)
- `rotation_error` — geodesic angle between estimated and ground-truth rotation
- `translation_error` — angular error between translation directions
- `epipolar_residual` — mean symmetric epipolar distance
- `fundamental_from_essential` — derives F from E and K
- `compute_metrics` — aggregates all metrics into a single dict

**`src/visualization.py`** — plotting utilities for all experiments
- `plot_image_pair`, `plot_degradation_gallery`, `plot_preprocessing_gallery`, `plot_matches`
- `plot_metric_vs_severity`, `plot_preprocessing_comparison`, `plot_degradation_overview`, `plot_model_comparison`
- `save_fig` — saves figure to `results/`

## Dataset

Uses the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
Specific sequence used in Experiments: `fr3/long_office_household` 
Place the extracted folder at `data/DATASET_PATH/`

## Experiments

| # | Question |
|---|---|
| 1 | How do degradation type and severity affect match count, inlier ratio, and pose error? |
| 2 | Which preprocessing method best recovers accuracy under each degradation? |
| 3 | Is Essential matrix or Homography more robust to image corruption? |

## Dependencies

```
numpy  opencv-python  scipy  matplotlib  scikit-image  tqdm
```

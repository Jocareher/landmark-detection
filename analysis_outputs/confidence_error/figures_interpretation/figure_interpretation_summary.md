# Figure Interpretation Summary

## Data and Protocol Checks
- Source directory: `/Users/jocareher/Downloads/confidence_error_babyland72`
- Output directory: `/Users/jocareher/Library/CloudStorage/OneDrive-Personal/Educacion/PhD_UPF_2023/landmarks_detection/analysis_outputs/confidence_error/figures_interpretation`
- Per-image rows: 622
- Evaluable landmark rows: 30008
- Mean official per-image NME: 0.1041 fraction / 10.41%
- Invalid GT landmarks are excluded from error computations through the corrected official evaluation mask.
- Predicted visibility is treated only as an analysis variable, not as the official mask.
- NME values stored as fractions were multiplied by 100 for plotting.

## Available CSV Schemas
- `confidence_error_correlations.csv`: confidence_signal, n, pearson, scope, spearman
- `confidence_quantile_errors.csv`: confidence_signal, mean_nme, median_nme, quantile, retained_landmarks
- `failure_detection.csv`: auprc, auroc, confidence_selection, confidence_signal, failure_threshold, n, precision_good, recall_good
- `per_image_confidence_error.csv`: box_normalization_factor, image_id, image_path, max_nme, max_nme_percent, mean_heatmap_entropy, mean_heatmap_max, mean_nme, mean_nme_percent, mean_pca_reconstruction_error, mean_tta_variance, median_nme, median_nme_percent, number_of_evaluable_landmarks, number_of_invalid_landmarks, number_of_nan_target_landmarks, number_of_valid_landmarks, pose, total_landmarks
- `per_landmark_confidence_error.csv`: evaluable_for_error, grouped_region, gt_valid_for_error, heatmap_entropy, heatmap_max, heatmap_variance, image_id, image_path, landmark_index, normalized_error, normalized_error_percent, pca_reconstruction_error, peak_sharpness, pixel_error, pose, predicted_visibility, predicted_x, predicted_y, prediction_id, prediction_is_finite, region, target_is_finite, target_x, target_y, tta_variance, visibility
- `region_pseudo_label_viability.csv`: best_signal_at_25pct, failure_rate, mean_nme, median_nme, recommendation, region, retained_fraction, retained_landmarks, suitable_for_pseudo_labeling
- `retention_curves.csv`: confidence_signal, failure_rate, mean_nme, median_nme, region, retained_fraction, retained_landmarks
- `summary_by_region.csv`: invalid_gt_landmark_count, landmark_count, mean_heatmap_entropy, mean_heatmap_max, mean_nme, mean_nme_percent, mean_tta_variance, median_nme, median_nme_percent, region, spearman_entropy_vs_error, spearman_heatmap_max_vs_error, spearman_tta_variance_vs_error, total_rows, valid_gt_landmark_count

## Figure-by-Figure Interpretation
### `fig01_nme_by_pose`
- What it shows / construction: Mean per-image official NME by pose with image counts.
- Conclusion supported: The official per-image NME is 10.41%; profiles are the highest-risk poses.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig02_region_nme_and_validity`
- What it shows / construction: Two aligned panels: grouped-region NME and valid/invalid GT counts.
- Conclusion supported: Contour has high error and many invalid/excluded GT landmarks, which weakens it as an early pseudo-label target.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig03_confidence_signal_correlation_ranking`
- What it shows / construction: Global Spearman correlations; blue signals should decrease with error, orange uncertainty signals should increase.
- Conclusion supported: Heatmap variance, TTA variance, and heatmap max have the strongest monotonic relation with error; peak sharpness is weak.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig04_failure_detection_auroc`
- What it shows / construction: AUROC for failure detection at NME > 0.05.
- Conclusion supported: Heatmap variance and heatmap max are the best failure detectors; peak sharpness is close to random.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig05_retention_curves_strong_signals`
- What it shows / construction: Mean retained NME versus retained landmark fraction for strong interpretable signals.
- Conclusion supported: Confidence filtering clearly reduces expected pseudo-label noise at strict retained fractions.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig06_region_pseudo_label_viability`
- What it shows / construction: Annotated table using the region pseudo-label viability CSV.
- Conclusion supported: Mouth is the only early region; eyes and nose need strict/later use; contour and eyebrows should be delayed or excluded.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig07_top25_retained_nme_by_region`
- What it shows / construction: Top-25 retained NME per region using each region's best signal.
- Conclusion supported: Mouth has the cleanest retained subset; contour and eyebrows remain too noisy after filtering.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig08_pose_pseudo_label_risk`
- What it shows / construction: Pose-stratified top-25 filtering with low heatmap variance.
- Conclusion supported: Filtering helps, but profile poses remain higher risk and should be introduced cautiously.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.
### `fig09_heatmap_variance_vs_error_density`
- What it shows / construction: Density plot of heatmap variance against landmark NME with binned median trend.
- Conclusion supported: Higher heatmap variance is associated with higher error.
- Caveat: Density plots show association, not perfect calibration; high-confidence failures can still occur.
### `fig10_tta_variance_vs_error_density`
- What it shows / construction: Density plot of TTA variance against landmark NME with binned median trend.
- Conclusion supported: TTA variance behaves as an uncertainty signal and supports conservative filtering.
- Caveat: Density plots show association, not perfect calibration; high-confidence failures can still occur.
### `fig11_heatmap_max_vs_error_density`
- What it shows / construction: Density plot of heatmap maximum against landmark NME with binned median trend.
- Conclusion supported: Higher heatmap max is generally associated with lower error, but high-confidence failures remain possible.
- Caveat: Density plots show association, not perfect calibration; high-confidence failures can still occur.
### `fig12_pca_error_image_level_analysis`
- What it shows / construction: Image-level PCA diagnostic rather than a landmark-level signal.
- Conclusion supported: PCA reconstruction error is unavailable in this run and should not guide pseudo-label selection.
- Caveat: PCA reconstruction error is image-level and unavailable in this run; it must not be interpreted as a landmark-level confidence signal.
### `fig13_uda_strategy_recommendation`
- What it shows / construction: Compact decision matrix translating diagnostics into UDA choices.
- Conclusion supported: Recommended next step is consistency plus very conservative region-specific pseudo-labeling, beginning with mouth only.
- Caveat: Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation.

## Direct Answers
1. **Strongest confidence signals:** heatmap variance is strongest overall, followed closely by TTA variance and heatmap max. Heatmap entropy is useful but weaker. Peak sharpness is weak and should not drive selection.
2. **Safest pseudo-labeling regions:** mouth is the safest early region. Eyes and nose may be used later with strict filtering.
3. **Regions to exclude initially:** contour and eyebrows should be excluded or delayed. Contour is especially risky because it combines high NME with many invalid/excluded GT landmarks.
4. **Does filtering reduce pseudo-label noise?** Yes. Retention curves and top-25 summaries show lower retained NME under strong signals, but filtering is not sufficient for all regions.
5. **First UDA experiment:** start with consistency training plus very conservative pseudo-labeling for mouth only, or run consistency-only as a baseline if the experiment budget allows. Avoid broad all-landmark pseudo-labeling.
6. **PCA reconstruction error:** treat it as an image-level shape plausibility diagnostic only. In these corrected outputs it is unavailable because all values are NaN, so it should not influence pseudo-label selection.

## Generated Files
- `fig01_nme_by_pose.png`
- `fig01_nme_by_pose.pdf`
- `fig02_region_nme_and_validity.png`
- `fig02_region_nme_and_validity.pdf`
- `fig03_confidence_signal_correlation_ranking.png`
- `fig03_confidence_signal_correlation_ranking.pdf`
- `fig04_failure_detection_auroc.png`
- `fig04_failure_detection_auroc.pdf`
- `fig05_retention_curves_strong_signals.png`
- `fig05_retention_curves_strong_signals.pdf`
- `fig06_region_pseudo_label_viability.png`
- `fig06_region_pseudo_label_viability.pdf`
- `fig07_top25_retained_nme_by_region.png`
- `fig07_top25_retained_nme_by_region.pdf`
- `fig08_pose_pseudo_label_risk.png`
- `fig08_pose_pseudo_label_risk.pdf`
- `fig09_heatmap_variance_vs_error_density.png`
- `fig09_heatmap_variance_vs_error_density.pdf`
- `fig10_tta_variance_vs_error_density.png`
- `fig10_tta_variance_vs_error_density.pdf`
- `fig11_heatmap_max_vs_error_density.png`
- `fig11_heatmap_max_vs_error_density.pdf`
- `fig12_pca_error_image_level_analysis.png`
- `fig12_pca_error_image_level_analysis.pdf`
- `fig13_uda_strategy_recommendation.png`
- `fig13_uda_strategy_recommendation.pdf`

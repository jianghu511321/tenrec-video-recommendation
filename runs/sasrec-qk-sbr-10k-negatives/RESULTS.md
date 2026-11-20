# SASRec Negative-Sampling Comparison

**English** | [简体中文](结果总结.md)

This experiment compares two negative-sampling methods on Tenrec's SBR derivative of QK-video. A negative sample is a video used as a comparison against an actual click during training. With click-only data, an unclicked video may still interest the user.

## Data and evaluation protocol

- The first 10,000 user groups contain 463,834 clicks. The experiment retains 9,919 users with at least five clicks each.
- Each user's final two clicks serve as validation and test targets. The earlier 443,782 clicks provide 433,863 next-item training examples. All 135,918 candidate videos come from training histories.
- Validation and test are filtered independently. Eligible targets are first-time clicks on videos in the training catalog. Validation retains 7,808 targets, excluding 53 repeat clicks and 2,058 out-of-catalog targets. Test retains 7,860 targets, excluding 85 repeat clicks and 1,974 out-of-catalog targets.
- At the test recommendation point, the penultimate click has already occurred and may be used as input. For 1,324 test histories, that click is outside the training vocabulary and is skipped in the model input.

## Comparison settings

The popularity baseline ranks videos by training click count. Both SASRec variants are trained from scratch for eight epochs, with seed 20261114, a learning rate of 0.001, and 64-dimensional video embeddings. Training uses causal sequence windows, predicting up to 25 positions per window. Each prediction only sees earlier clicks. FAISS performs exact inner-product retrieval over the same full candidate catalog, and previously clicked videos are filtered.

The uniform variant samples uniformly from the catalog. The mixed variant chooses uniform sampling with 50% probability and sampling proportional to training click counts with 50% probability for each example. Both use one negative per training target and exclude items in that user's training history. Checkpoints are selected separately by validation NDCG@20. Constructing individual prefixes proved expensive on CPU, so both comparisons were restarted using the same sequence-window implementation. The `uniform/` directory contains the interrupted first epoch from the earlier implementation. The table uses `uniform-windowed/` and `mixed-windowed/`.

## Measured results

| Method | Selected epoch | Validation NDCG@20 | Test Recall@20 | Test Recall@100 | Test NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Popularity | — | 0.010929 | 2.3410% (184 hits) | 7.2901% (573 hits) | 0.009310 |
| SASRec, uniform | 8 | 0.007508 | 1.6921% (133 hits) | 5.4707% (430 hits) | 0.006979 |
| SASRec, mixed | 5 | 0.000163 | 0.0127% (1 hit) | 0.0763% (6 hits) | 0.0000355 |

The 50/50 mixture reduced both top-20 performance and top-100 coverage. Validation diagnostics showed that 76.1% of the mixed variant's top-20 recommendation slots contained videos with only one or two training clicks, compared with 0.16% for uniform sampling. The median training click count was 14 for validation targets and 1 for videos recommended by the mixed model. One possible explanation is that popular videos became negative examples too frequently, pushing their scores down excessively. This explanation needs an independent comparison of sampling proportions or difficulty.

The findings apply to this user cohort, seed, and eight-epoch budget. The uniform variant's validation score was still rising in its last epoch. The experiment reused a user cohort whose earlier test results had been inspected, so the test scores are exploratory. The catalog and eligibility rules changed substantially from the earlier experiment. Method comparisons should use the common protocol within this table.

Reproduction commands are in the [English README](../../README.md). Dataset counts are in [data/manifest.json](data/manifest.json). Per-epoch validation records and final evaluations are saved under the respective run directories. Checkpoints are kept locally. Recommendation-frequency diagnostics are in [validation_catalog_diagnostics.json](validation_catalog_diagnostics.json).

# Tenrec Similar-User Retrieval Experiment

**English** | [简体中文](结果总结.md)

On a separate user cohort, similar-user retrieval increased NDCG@20 from the popularity baseline's 0.01032 to 0.04204, approximately 4.07 times the baseline. The number of users whose target appeared in the top 20 increased from 213 to 709.

## Method

The model uses memory-based collaborative filtering. Earlier click histories from 9,919 users form a sparse user–video matrix, with each row recording the videos clicked by a training user. The candidate catalog contains the 135,918 videos found in these training histories.

Given a user's history, the model gives higher weights to recent clicks and to shared videos clicked by fewer people. A shared rare-video interaction can distinguish interests more strongly than a shared popular-video interaction. Cosine similarity compares the weighted query history with training users, and up to 160 nearest neighbors are selected. Each neighbor votes for the videos they clicked, with a weight equal to squared similarity. Previously clicked videos are filtered, and candidates are ranked by votes. Popularity fills short candidate lists.

Validation compared 64 configurations on the original 7,808 validation targets, varying neighbor counts, rare-video weighting, recency, and popularity rank fusion. The selected configuration achieved validation NDCG@20 of 0.03755, compared with 0.01093 for popularity. Ordinary binary-history similarity with the same 160 neighbors achieved 0.03370. This difference measures the combined effect of the two weighting changes.

## Results

| User cohort | Method | Recall@20 | Recall@100 | NDCG@20 |
|---|---|---:|---:|---:|
| Separate cohort, 7,947 users | Popularity | 2.68% | 8.02% | 0.01032 |
| Separate cohort, 7,947 users | Similar-user retrieval | **8.92%** | **18.82%** | **0.04204** |
| Previous test cohort, 7,860 users | Popularity | 2.34% | 7.29% | 0.00931 |
| Previous test cohort, 7,860 users | Similar-user retrieval | **8.22%** | **18.28%** | **0.03706** |

Recall@20 measures the fraction of users whose actual next click appears in the top 20. NDCG@20 also accounts for the hit's rank, rewarding earlier positions. On the separate cohort, the NDCG@20 difference was 0.03172. A paired bootstrap over the same users, with 2,000 resamples, produced a 95% interval of approximately [0.02862, 0.03464]. This interval describes sampling variability within this evaluation cohort.

## Evaluation scope

After selecting parameters on validation, the experiment evaluated user groups 10,001 through 20,000 in the source file. The two cohorts were checked for overlapping users. Earlier clicks from the separate cohort served only as query histories. Both methods used the same training catalog, evaluation users, and previously-clicked-item filter. Each user's final click was the target, with eligibility requiring a first-time click on a video present in the training catalog. Of 10,000 user groups, the exclusions were 1,915 out-of-catalog targets, 64 repeated targets, 71 users with insufficient clicks, and 3 users with no known video in their history. The previous test cohort provides continuity with earlier experiments. The separate cohort is the primary evaluation for this run.

The SBR file contains click data already filtered by Tenrec. This experiment uses two cohorts taken in source-file order, with temporal order inherited from each user's event sequence. The findings apply to this warm-start video task and the popularity baseline. Other datasets, full-scale runs, and online traffic require further evaluation.

Reproduction commands are in the [English README](../../README.md). Validation parameters and scores are in [selection.json](selection.json). Evaluation metrics and sample counts are in [evaluation.json](evaluation.json).

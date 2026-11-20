# Tenrec QK-video Sequential Retrieval Pilot

**English** | [简体中文](结果总结.md)

## Objective and scope

This initial experiment explored SASRec sequential retrieval with FAISS for the video recommendation project. It used the first 10,000 grouped user histories from Tenrec's official `sbr_data_1M.csv` derivative. The derivative contains click-positive sequences and applies Tenrec's `gen_sbr.py` filtering, removing videos with fewer than two or more than 200,000 corpus clicks. The selected cohort defines this pilot's scale.

Raw QK-video user blocks were found to recur in different parts of the file. Taking a raw CSV prefix could therefore truncate user histories. The pilot switched to the official user-grouped SBR file and checked for repeated user-history groups. The source download remains local under `E:\Tenrec\Tenrec`.

## Split and model

- The selected 10,000 user groups contained 463,834 clicks. Users needed at least five clicks to form a training prefix and two held-out targets.
- Within each user's source order, all but the final two clicks formed the training history. The penultimate click was the validation target, and the final click was the test target.
- Candidates were restricted to videos found in the selected cohort's training histories. Both held-out targets had to occur in the training catalog. The final filtered cohort contained 5,642 users and 77,057 videos, producing 234,953 next-item training pairs.
- The model used one causal Transformer layer, 64-dimensional item embeddings, up to 50 historical clicks, one sampled negative per positive, and a pairwise softplus ranking loss. FAISS `IndexFlatIP` performed exact inner-product retrieval across all 77,057 warm-catalog videos, followed by previously-clicked-item filtering.
- Training ran for five epochs, followed by five continuation epochs from the epoch-5 weights. The continuation used a fresh Adam state and lowered the learning rate from 0.001 to 0.0005. Validation NDCG@20 selected epoch 9. The random seed was 20261114, and training used CPU.
- Negative sampling excluded training-history videos. A future positive could still be sampled as a negative, introducing label noise in the implicit-feedback task. Future held-out labels were reserved for evaluation.

## Results

| Method | Validation NDCG@20 | Test Recall@20 | Test NDCG@20 |
|---|---:|---:|---:|
| Training-item popularity | 0.01275 | 0.02783 | 0.01127 |
| SASRec + FAISS | 0.01114 | 0.02233 | 0.00943 |

SASRec's test NDCG@20 was lower by 0.00184, approximately 16.3% relative to popularity. Recall@20 was lower by 0.00549, approximately 19.7%. Popularity performed better in this cohort and configuration despite the sequential model's decreasing training loss.

Each test user had one held-out target, making Recall@20 equal to HitRate@20. NDCG@20 also rewards higher-ranked hits. The metrics describe next-click retrieval among the 77,057 training-catalog videos under this offline protocol.

## Test-set history

The initial five-epoch test result was inspected before continuing training based on the validation curve. Continuation checkpoints were selected using validation NDCG@20. Since the test scores had already been seen, the reported results are exploratory. A new held-out user cohort or time period would be needed for a fresh final evaluation.

This pilot used one seed and one 10,000-user prefix cohort. At that stage, multi-seed evaluation, other user cohorts, SASRec ablations, LightGCN, DeepFM/DCNv2 ranking, four-task MMoE, and the broader planned model comparisons remained additional work. This report records the initial sequential retrieval pilot. Later experiments are documented in the [negative-sampling report](runs/sasrec-qk-sbr-10k-negatives/RESULTS.md) and the [similar-user retrieval report](runs/tenrec-neighbor-retrieval/RESULTS.md).

## Runtime and reproducibility limits

The machine had an RTX 3060 Laptop GPU with 6 GB of memory, but the available PyTorch build was `2.8.0+cpu`, with CUDA reported as unavailable. Free system RAM was approximately 3.2 GB when checked. Raw QK-video occupied about 15.1 GB. The pilot therefore read a small cohort from the 1.12 GB SBR derivative. Full-scale training would require streaming or chunked training and indexing, plus a suitable CUDA environment or more RAM.

## Artifacts

- [Original manifest](runs/sasrec-qk-sbr-10k/manifest.json) records source metadata, splits, and counts.
- The prepared `runs/sasrec-qk-sbr-10k/cohort.json` remains local.
- [Initial validation records](runs/sasrec-qk-sbr-10k/validation_by_epoch.json) contain epochs 1–5.
- [Popularity baseline](runs/sasrec-qk-sbr-10k/popularity_baseline.json) records the reference scores.
- [Continuation validation records](runs/sasrec-qk-sbr-10k-continue/validation_by_epoch.json) cover epochs 5–10.
- [Continuation evaluation](runs/sasrec-qk-sbr-10k-continue/evaluation.json) contains the selected checkpoint's test metrics.
- The selected `runs/sasrec-qk-sbr-10k-continue/best_sasrec.pt` checkpoint remains local.

[Project README](README.md)

# Tenrec Video Recommendation

Offline next-click video recommendation on a subset of the Tenrec QK-video SBR dataset. The project compares popularity, SASRec-style sequential retrieval with FAISS, negative-sampling variants, and similarity-weighted collaborative retrieval.

The best measured method uses recent clicks and shared rare-video interactions to find similar users. On a separate cohort of 7,947 eligible users, it increased Recall@20 from **2.68% to 8.92%** and NDCG@20 from **0.01032 to 0.04204** against training-item popularity.

## Results

The collaborative retriever was fitted on the earlier click histories of 9,919 users. The training catalog contains 135,918 videos. Its configuration was selected on 7,808 validation targets, then evaluated on the next 10,000 user groups in the source file. Of those, 7,947 had eligible next-click targets and usable history.

| Method | Recall@20 | Recall@100 | NDCG@20 |
|---|---:|---:|---:|
| Popularity | 2.68% | 8.02% | 0.01032 |
| Similar-user retrieval | **8.92%** | **18.82%** | **0.04204** |

The NDCG@20 difference is 0.03172. A paired bootstrap over users, with 2,000 resamples, gives a 95% interval of [0.02862, 0.03464]. Both methods use the same catalog, evaluation users, and previously-clicked-item filter.

[Detailed report (Chinese)](runs/tenrec-neighbor-retrieval/结果总结.md) · [Evaluation JSON](runs/tenrec-neighbor-retrieval/evaluation.json) · [Validation search](runs/tenrec-neighbor-retrieval/selection.json)

## How the collaborative retriever works

1. Build a sparse binary user–video matrix using training clicks.
2. Weight a query user's clicks by recency and the rarity of shared videos.
3. Compute cosine similarity to training users and select up to 160 neighbors.
4. Give each neighbor's clicked videos a vote weighted by squared similarity.
5. Filter previously clicked videos, return the top 100, and fill short lists from popularity.

Validation compared 64 configurations covering neighbor counts, rarity weighting, recency, popularity penalties, and rank fusion. The selected configuration is recorded before the independent user-cohort evaluation.

## Setup

The recorded runs used Windows, Python 3.9.12, PyTorch 2.8.0+cpu, NumPy 1.26.4, SciPy 1.13.1, and faiss-cpu 1.13.0. CPU runs are supported. SASRec also accepts a CUDA device when a suitable PyTorch build is installed.

```powershell
git clone https://github.com/jianghu511321/tenrec-video-recommendation.git
cd tenrec-video-recommendation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data and reproducibility

Download Tenrec from its [official dataset portal](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html) and accept its license agreement. Use `sbr_data_1M.csv`, the QK-video click-history derivative supplied for sequential recommendation. The local source file used here is 1,120,679,998 bytes.

The repository includes source code, configurations, aggregate metrics, and reports. Dataset files, user/item mappings, interaction cohorts, checkpoints, Python environments, and reference-source snapshots are excluded from Git. Scripts regenerate the cohort and fitted representations from the source dataset. Upstream references and commits are in [SOURCES.md](SOURCES.md).

Run the following from the repository root with a fresh output directory. Set `$tenrecSource` to your downloaded file.

```powershell
$experimentPython = '.\.venv\Scripts\python.exe'
$tenrecSource = 'E:\Tenrec\Tenrec\sbr_data_1M.csv'
$experimentRoot = '.\runs\neighbor-repro'

& $experimentPython .\prepare_sasrec_cohort.py --source $tenrecSource --max-users 10000 --out "$experimentRoot\data"
& $experimentPython .\train_neighbor_retrieval.py develop --cohort "$experimentRoot\data\cohort.json" --out "$experimentRoot\model"
& $experimentPython .\train_neighbor_retrieval.py evaluate --cohort "$experimentRoot\data\cohort.json" --out "$experimentRoot\model" --source $tenrecSource --holdout-start 10000 --holdout-users 10000
```

Each development user contributes all but their final two clicks to training. The penultimate click is the validation target. Candidate videos are defined entirely by training clicks. Evaluation requires a first-time target in that catalog. The independent cohort contributes its observed history as a query and holds out its final click.

## SASRec and negative-sampling comparison

The SASRec-style model uses a one-layer causal Transformer, 64-dimensional item vectors, histories of up to 50 clicks, and a sampled-negative pairwise loss. FAISS `IndexFlatIP` performs exact inner-product retrieval over the training catalog.

The following comparison uses the corrected original-user test cohort of 7,860 users. It has a different evaluation population from the independent-cohort table above.

| Method | Recall@20 | Recall@100 | NDCG@20 |
|---|---:|---:|---:|
| Popularity | 2.34% | 7.29% | 0.009310 |
| SASRec, uniform negatives | 1.69% | 5.47% | 0.006979 |
| SASRec, 50/50 uniform and popularity-weighted negatives | 0.013% | 0.076% | 0.0000355 |

[Negative-sampling report (Chinese)](runs/sasrec-qk-sbr-10k-negatives/结果总结.md)

To reproduce these comparisons after preparing a fresh cohort as above, run:

```powershell
& $experimentPython .\baseline_popularity.py --data "$experimentRoot\data" --out "$experimentRoot\popularity"
foreach ($experimentSampler in @('uniform', 'mixed')) {
    & $experimentPython .\train_sasrec_faiss.py --data "$experimentRoot\data" --out "$experimentRoot\$experimentSampler-windowed" --epochs 8 --batch-size 128 --max-len 50 --embedding-dim 64 --learning-rate 0.001 --device cpu --cpu-threads 4 --seed 20261114 --negative-sampling $experimentSampler --sequence-batches
}
```

## Project files

| File | Purpose |
|---|---|
| `prepare_sasrec_cohort.py` | Encode training histories and prepare chronological evaluation targets |
| `train_neighbor_retrieval.py` | Build the collaborative index, select parameters, and evaluate separate users |
| `train_sasrec_faiss.py` | Train SASRec-style retrieval with uniform or mixed negative sampling |
| `evaluate_sasrec_faiss.py` | Load a saved sequential model and evaluate retrieval |
| `baseline_popularity.py` | Evaluate training-item popularity |
| `configs/` | Recorded experiment settings |
| `runs/` | Aggregate metrics, manifests, validation records, and reports |

## Evaluation scope

These are offline experiments on selected user cohorts from the official one-million-user SBR derivative. The fitted development cohort has 9,919 users and a 135,918-video catalog. The source data is click-positive and already filtered by Tenrec's preprocessing. User-level event ordering comes from the supplied file. Reported metrics apply to eligible targets in the training catalog.

The independent cohort was evaluated after validation selected the collaborative configuration. Earlier SASRec iterations reused an exploratory test cohort. Run reports document the respective protocols, exclusions, and failed experiments. Scaling to all users and videos, exposure-level CTR ranking, and online performance require additional experiments.

# Tenrec 视频推荐

[English](README.md) | **简体中文**

本项目在 Tenrec QK-video 派生的 SBR 数据子集上进行离线下一次点击推荐，对比热门度、SASRec 风格序列召回与 FAISS 检索、不同负采样方法，以及基于相似用户加权投票的协同召回。

目前实测表现最好的方法通过近期点击和共同点击的冷门视频寻找相似用户。在另一批 **7,947 名**符合评估条件的用户上，相比训练集热门度基线，Recall@20 从 **2.68% 提升到 8.92%**，NDCG@20 从 **0.01032 提升到 0.04204**。

## 实验结果

协同召回模型使用 9,919 名用户的早期点击历史建立索引，训练候选库包含 135,918 个视频。在 7,808 个验证目标上选定配置后，再评估源文件中接下来的 10,000 个用户组，其中 7,947 人的下一次点击目标和历史符合评估条件。

| 方法 | Recall@20 | Recall@100 | NDCG@20 |
|---|---:|---:|---:|
| 热门度 | 2.68% | 8.02% | 0.01032 |
| 相似用户召回 | **8.92%** | **18.82%** | **0.04204** |

NDCG@20 的差值为 0.03172。对用户进行 2,000 次配对 bootstrap 重抽样，得到的 95% 区间为 [0.02862, 0.03464]。两种方法使用相同的候选库、评估用户和已点击视频过滤规则。

[详细结果报告](runs/tenrec-neighbor-retrieval/结果总结.md) · [评估 JSON](runs/tenrec-neighbor-retrieval/evaluation.json) · [验证参数搜索](runs/tenrec-neighbor-retrieval/selection.json)

## 协同召回流程

1. 用训练点击建立稀疏的用户与视频二值交互矩阵。
2. 根据点击的新近程度和共同视频的稀有程度，对查询用户的历史加权。
3. 计算查询用户与训练用户的余弦相似度，最多选择 160 名邻居。
4. 邻居对自己点击过的视频投票，票重为相似度的平方。
5. 过滤查询用户已点击的视频，返回前 100 个候选，数量不足时用热门榜补齐。

验证阶段比较了 64 组配置，覆盖邻居数量、稀有视频权重、近期点击权重、热门度惩罚和名次融合。新用户批次评估前，选出的配置已保存。

## 环境安装

实验记录使用 Windows、Python 3.9.12、PyTorch 2.8.0+cpu、NumPy 1.26.4、SciPy 1.13.1 和 faiss-cpu 1.13.0。项目支持 CPU 运行。安装支持 CUDA 的 PyTorch 后，SASRec 也可以指定 CUDA 设备。

```powershell
git clone https://github.com/jianghu511321/tenrec-video-recommendation.git
cd tenrec-video-recommendation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 数据和复现

从 [Tenrec 官方数据入口](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html)下载数据并接受许可协议，使用供序列推荐任务使用的 QK-video 点击历史派生文件 `sbr_data_1M.csv`。本地实验使用的文件大小为 1,120,679,998 字节。

仓库提供源码、配置、汇总指标和报告。数据文件、用户与视频 ID 映射、交互 cohort、模型权重、Python 环境和参考源码副本通过 Git 忽略规则保留在本地。脚本可以从原始数据重新生成 cohort 和模型所需的表示。上游来源与提交版本见 [SOURCES.md](SOURCES.md)。

在仓库根目录运行以下命令，使用新的输出目录，并将 `$tenrecSource` 改为下载文件的路径。

```powershell
$experimentPython = '.\.venv\Scripts\python.exe'
$tenrecSource = 'E:\Tenrec\Tenrec\sbr_data_1M.csv'
$experimentRoot = '.\runs\neighbor-repro'

& $experimentPython .\prepare_sasrec_cohort.py --source $tenrecSource --max-users 10000 --out "$experimentRoot\data"
& $experimentPython .\train_neighbor_retrieval.py develop --cohort "$experimentRoot\data\cohort.json" --out "$experimentRoot\model"
& $experimentPython .\train_neighbor_retrieval.py evaluate --cohort "$experimentRoot\data\cohort.json" --out "$experimentRoot\model" --source $tenrecSource --holdout-start 10000 --holdout-users 10000
```

开发批次中，每个用户最后两次点击之前的记录用于训练，倒数第二次点击作为验证目标。候选视频完全由训练点击确定。评估要求目标视频属于候选库，而且用户此前没有点击过。独立批次的已发生历史用作查询输入，最后一次点击作为留出目标。

## SASRec 和负采样对照

SASRec 风格模型使用一层带因果遮罩的 Transformer、64 维视频向量、最多 50 次历史点击，以及基于采样负例的成对排序损失。FAISS `IndexFlatIP` 在训练候选库中执行精确内积检索。

下表使用修正规则后原用户批次中的 7,860 名测试用户，与上方独立用户批次表格的评估人群不同。

| 方法 | Recall@20 | Recall@100 | NDCG@20 |
|---|---:|---:|---:|
| 热门度 | 2.34% | 7.29% | 0.009310 |
| SASRec，均匀随机负采样 | 1.69% | 5.47% | 0.006979 |
| SASRec，均匀与热门度加权各占一半的混合负采样 | 0.013% | 0.076% | 0.0000355 |

[负采样结果报告](runs/sasrec-qk-sbr-10k-negatives/结果总结.md)

按上面的步骤准备新的 cohort 后，可运行以下命令复现对照。

```powershell
& $experimentPython .\baseline_popularity.py --data "$experimentRoot\data" --out "$experimentRoot\popularity"
foreach ($experimentSampler in @('uniform', 'mixed')) {
    & $experimentPython .\train_sasrec_faiss.py --data "$experimentRoot\data" --out "$experimentRoot\$experimentSampler-windowed" --epochs 8 --batch-size 128 --max-len 50 --embedding-dim 64 --learning-rate 0.001 --device cpu --cpu-threads 4 --seed 20261114 --negative-sampling $experimentSampler --sequence-batches
}
```

[初始序列召回实验报告](结果总结.md)

## 项目文件

| 文件 | 用途 |
|---|---|
| `prepare_sasrec_cohort.py` | 编码训练历史，按用户内时间顺序准备评估目标 |
| `train_neighbor_retrieval.py` | 建立协同索引、选择参数并评估另一批用户 |
| `train_sasrec_faiss.py` | 使用均匀或混合负采样训练 SASRec 风格召回模型 |
| `evaluate_sasrec_faiss.py` | 加载保存的序列模型，评估召回效果 |
| `baseline_popularity.py` | 评估训练集热门度基线 |
| `configs/` | 已记录的实验配置 |
| `runs/` | 汇总指标、数据清单、验证记录和报告 |

## 评估范围

这些离线实验使用官方百万用户 SBR 派生文件中选取的用户批次。拟合模型的开发批次有 9,919 名用户和 135,918 个候选视频。源文件包含点击正反馈，已经过 Tenrec 的预处理。用户内的时间顺序来自文件记录顺序，指标适用于训练候选库中符合条件的目标。

协同配置在验证集上选定后，才评估独立用户批次。较早的 SASRec 迭代复用了探索性测试批次。各实验报告记录了划分协议、排除条件和失败实验。全量用户与视频规模、曝光级 CTR 排序，以及线上表现需要进一步实验。

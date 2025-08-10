# Jarir B2B Recommendation System (Two-Stage: Retriever + Reranker)

End-to-end system that recommends relevant SKUs to Jarir B2B customers using a modern two-stage architecture:
- Retrieval: learned Two-Tower embeddings to quickly narrow a large catalog
- Ranking: MLP reranker with rich features to optimize top-K precision

This README explains how to set up, run, and evaluate the full pipeline; where outputs are saved; and how to take the system toward production.

## Project layout

```
KAUST-Project/
  data/
    raw/                    # Place raw files here (e.g., jarir.xlsx)
    processed/jarir/        # All processed tables and model artifacts
  notebooks/
    01_data_prep.ipynb      # Clean data, build interactions, sequences & splits
    02_embeddings_and_baselines.ipynb  # Build matrices, run baselines
    03_retriever_and_index.ipynb       # Train Two-Tower, export embeddings
    04_ranker_and_eval.ipynb           # Candidates, features, train reranker, metrics
  scripts/
    create_interaction_matrices.py
  src/
    ingestion/ETL.py        # Optional CSV ETL (customers/products/transactions)
    models/interaction_matrix.py
  README.md
  requirements.txt
```

## Environment

- Python 3.10+ recommended
- GPU optional (speedup for training/feature building). If using CUDA on Windows, ensure compatible drivers and versions
- Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- If you do not have a compatible CUDA, you may remove/replace `faiss-gpu-cu12` with CPU FAISS or skip FAISS entirely (FAISS is not required for the notebooks).
- Parquet engines: this repo uses `pyarrow` and `fastparquet`.

## Data

- Expected raw input: `data/raw/jarir.xlsx` (Jarir transactions). One row roughly equals a purchased line.
- Optional ETL (for CSV-based sources): see `src/ingestion/ETL.py`.

## How to run (end-to-end)

1) 01_data_prep.ipynb
- Cleans and normalizes transactions
- Computes robust price, removes non-product lines and outliers
- Builds time-ordered next-item sequences and 80/10/10 splits (train/val/test)
- Writes:
  - `data/processed/jarir/interactions_clean.parquet`
  - `data/processed/jarir/items_clean.parquet`
  - `data/processed/jarir/customers_clean.parquet`
  - `data/processed/jarir/{item_id_map,customer_id_map}.parquet`
  - `data/processed/jarir/sequences_{train,val,test}.parquet`

2) 02_embeddings_and_baselines.ipynb
- Builds sparse matrices for user–item interactions and KNN
- Trains and evaluates baselines (Popularity, ItemKNN, UserKNN, NMF)
- Writes: `data/processed/jarir/baseline_results.json`

3) 03_retriever_and_index.ipynb
- Trains a Two-Tower retriever (user and item towers with residual blocks + LayerNorm)
- Saves learned embeddings
- Compares Recall@K vs baselines
- Writes:
  - `data/processed/jarir/user_embeddings.npy`
  - `data/processed/jarir/item_embeddings.npy`
  - `data/processed/jarir/twotower_results.json`

4) 04_ranker_and_eval.ipynb
- Generates candidates from Two-Tower embeddings
- Builds GPU features (dot(u,v), max-sim to recent history, popularity, hist length, price z-score)
- Trains an MLP reranker with early stopping; evaluates Recall@K, NDCG@K
- Computes Coverage@K on TEST and writes summaries
- Optional: K-Fold CV (timestamp-agnostic) and full retrain (train+val)
- Writes:
  - `data/processed/jarir/candidates_{train,val,test}.parquet`
  - `data/processed/jarir/ranker_feats_{split}_shards/part_*.parquet`
  - `data/processed/jarir/ranker_best.pt`
  - `data/processed/jarir/ranker_results.json`
  - `data/processed/jarir/summary_metrics.csv` (one row)
  - `data/processed/jarir/summary_notes.txt` (formatted deltas vs baseline)

## Metrics

- Recall@K: 1 if the target is in the top-K, else 0; averaged across queries
- NDCG@K: rank-sensitive quality (higher is better)
- Coverage@K: unique items recommended in top-K across users divided by catalog size (higher → broader exposure)

The ranker notebook prints VAL/TEST metrics and writes summaries for TEST:
- `summary_metrics.csv` (single row):
  - Columns: `Model,Recall@10,NDCG@10,Coverage@10,Notes`
  - Example model name: `Two-Stage Ranker`
- `summary_notes.txt` (bullet-style deltas vs ItemKNN):
  - `Two-Stage Ranker: best Recall@10 (X.XX); NDCG@10 (+Δ vs baseline).`
  - `- Two-Stage Ranker: Coverage@10 up to X%, Δ vs ItemKNN ±Y pp.`
  - Optional tie line if Recall@10 matches baseline

## Cross-validation (optional)

`04_ranker_and_eval.ipynb` includes a K-Fold CV cell (timestamp-agnostic) and a final full retrain on train+val:
- Reports per-fold Recall@K/NDCG@K and their means
- Saves a separate checkpoint for the full retrain and evaluates on TEST

## Reproducibility & configuration

- Seeds set for Python/NumPy/PyTorch; deterministic options toggled where supported
- Major hyperparameters centralized in `CFG` dicts inside notebooks (dimensions, batch sizes, learning rates, top-K sizes, sharding, regularization)

## Results (typical, example scale)

- Items ≈ 1,700; Users ≈ 900; Sequences: train ≈ 1,100 / val ≈ 170 / test ≈ 160
- Best baseline (ItemKNN) Recall@10 ≈ 0.11 (validation)
- Two-Tower Recall@10 ≈ 0.12 (validation)
- Reranker Recall@10 (within-candidate) ≈ 0.7 on dev slice (candidate size ~100)
- TEST summaries (Recall@10, NDCG@10, Coverage@10) are written by the final cell

Note: reranker metrics inside the notebook are “within-candidate”. For end-to-end KPI, combine retrieval coverage with rerank quality.

## Production notes

- ANN retrieval: index `item_embeddings.npy` with FAISS (HNSW/IVF) and serve top-K per user; rerank online with the MLP
- Cold start: combine popularity, brand/category priors, and nearest neighbors in embedding space
- Monitoring: track Recall/NDCG@K, Coverage@K, segment-level KPIs, and revenue-weighted variants; check drift for embeddings/features

## Troubleshooting

- Windows + CUDA: ensure versions match `torch` and (if used) FAISS; otherwise prefer CPU FAISS or skip FAISS entirely
- Parquet: install `pyarrow` and `fastparquet`. If one fails, the notebooks fall back where possible
- Memory: adjust `CFG["cand_batch"]`, `feat_batch_q`, and `batch_size` downward on limited-GPU/CPU machines

## License

TBD by project owners.

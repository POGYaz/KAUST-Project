    ### Jarir B2B Recommendation System — Architecture and Pipeline

    This document explains, end-to-end, what the system does from raw data to the final two-stage recommendations. It focuses on the “what” and the “why” of each component, the model internals, and how training/evaluation work.

    ---

    ### High-level overview

    - **Goal**: Recommend relevant SKUs to Jarir B2B customers based on past purchases and behavior.
    - **Approach**: A modern two-stage system
    - **Stage 1 — Retrieval (Two-Tower)**: Quickly narrows a large item catalog to a small candidate set (top-K per user).
    - **Stage 2 — Ranking (MLP Reranker)**: Reorders the candidates using richer features for higher precision.

    The pipeline is implemented primarily in four notebooks (01–04) and a few helper modules under `src/` and `scripts/`.

    ---

    ### Data and artifacts

    - Raw data: `data/raw/jarir.xlsx` (Jarir transactions; one row ≈ a purchased line).
    - Processed data outputs (notebook 01):
    - `data/processed/jarir/interactions_clean.parquet`: cleaned transactions
    - `data/processed/jarir/items_clean.parquet`: item catalog with stats
    - `data/processed/jarir/customers_clean.parquet`: customer table with stats
    - `data/processed/jarir/{item_id_map, customer_id_map}.parquet`: index maps
    - `data/processed/jarir/sequences_{train,val,test}.parquet`: supervised “next-item” samples per user over time
    - Baseline results (notebook 02): `data/processed/jarir/baseline_results.json`
    - Two-Tower outputs (notebook 03): `user_embeddings.npy`, `item_embeddings.npy`, `twotower_results.json`
    - Reranker outputs (notebook 04): `candidates_{train,val,test}.parquet`, sharded feature parquet under `ranker_feats_*`, `ranker_best.pt`, `ranker_results.json`

    ---

    ### Step 0 — Optional ETL (`src/ingestion/ETL.py`)

    - Reads normalized CSVs for customers, products, and transactions.
    - Casts types, validates SKU and CustomerID references, merges to a single table.
    - Saves CSV or Parquet depending on size.
    - This step is not mandatory when using `notebooks/01_data_prep.ipynb` directly on Excel.

    Why: Keep I/O, id normalization, and sanity checks centralized if data are provided as multiple tables.

    ---

    ### Step 1 — Data preparation (`notebooks/01_data_prep.ipynb`)

    Purpose: Clean raw purchase lines, normalize text, compute robust price, remove non-product lines and outliers, and build chronological sequences for supervised training.

    Key stages:
    - Configuration: thresholds/toggles for cleaning, seeds for reproducibility.
    - Load: read Excel (openpyxl engine), print shape and raw columns.
    - Date parsing: robustly parse Jarir-specific formats (e.g., `Jan-1`) and Excel serials; no synthetic timestamps if all parsing fails—an explicit error is raised.
    - Standardization: rename columns to a consistent schema (e.g., `stock_code`, `description`, `customer_id`, `invoice_date`, `quantity`, `line_amount`).
    - Price computation: `price = line_amount / quantity` with protection against division-by-zero and non-finite values.
    - Basic validity filters: drop rows missing core fields; remove returns (invoice prefix `C`); keep only positive quantity and price; drop exact duplicates.
    - Text normalization: Unicode NFKC, whitespace collapsing, uppercase; remove non-product lines by regex patterns.
    - Duplicate aggregation: group by (`customer_id`, `stock_code`, `invoice_date`) summing `quantity`/`line_amount`; recompute weighted price.
    - Outliers: log-scale IQR with configurable multiplier; winsorize or drop.
    - Coverage filters (optional): drop ultra-rare items/users.
    - Final tables: save `interactions_clean`, `item_catalog`, `customer_table` to Parquet.
    - Sequence building: construct next-item supervised samples per user with history window (HIST_MAX), and time-based 80/10/10 splits into train/val/test.

    Why: Supervised sequences turn raw transactions into training rows of the form: (user idx, history item idx list, positive item idx, timestamp).

    ---

    ### Step 2 — Baselines and interaction matrices (`notebooks/02_embeddings_and_baselines.ipynb`)

    Purpose: Establish performance baselines and build matrices to support KNN-style recommenders.

    - Build a user–item matrix from sequences where the positive target gets higher weight than history items (e.g., 1.0 vs 0.5), shape `[n_users, n_items]` (sparse CSR).
    - Build an auxiliary “KNN source” matrix from interactions (pre-cutoff) for better co-occurrence statistics.

    Baselines:
    - Popularity: ranks items by global frequency; masks seen items for each user.
    - ItemKNN: cosine similarity on item columns; user scores as profile·sim; masks seen items.
    - UserKNN: cosine similarity on user rows; aggregates top-k neighbors’ items weighted by similarity; masks seen items.
    - Matrix Factorization (NMF): factorizes user–item into W,H; scores as W[u]·H.

    Metrics:
    - Recall@K: 1 if the held-out positive appears in top-K, else 0; averaged.
    - NDCG@K, MRR@K: rank-sensitive metrics.

    Why: Baselines quantify “how hard the problem is” and provide sanity checks for later improvements.

    ---

    ### Step 3 — Two-Tower retriever (`notebooks/03_retriever_and_index.ipynb`)

    Purpose: Learn dense user and item embedding spaces such that a simple similarity retrieves relevant items fast from the full catalog.

    Model internals:
    - Embeddings: `user_emb: [n_users, d]`, `item_emb: [n_items, d]` with dimension `d = 256`.
    - Towers: each side passes through two residual blocks with `Linear → LayerNorm → ReLU → Dropout`, then a final `LayerNorm` and L2 normalization. Output vectors are unit-length.
    - Scoring: cosine similarity (dot product after L2 normalization).

    Training:
    - Objective: InfoNCE-style cross-entropy per example with one positive and K random negatives: maximize `sim(u, pos)` vs `sim(u, neg)`.
    - Temperature: scale logits by `1/temperature` to control softmax sharpness.
    - Negatives: sampled per-example from catalog excluding history + positive.
    - Optimization: AdamW; ReduceLROnPlateau scheduler monitoring dev Recall@K; early stopping.
    - Dev split: a slice from train by timestamp quantile; validation is reserved for final comparison.

    Outputs:
    - `user_embeddings.npy` and `item_embeddings.npy`: unit-normalized vectors for retrieval and features.
    - `twotower_results.json`: training curves and final dev/val metrics.

    Why: Embedding-based retrieval scales to large catalogs and enables ANN (e.g., FAISS) in production.

    ---

    ### Step 4 — Candidate generation (`notebooks/04_ranker_and_eval.ipynb`)

    Purpose: For each (user, context) construct a manageable set of candidates for reranking.

    How it works here:
    - Construct user vectors from recent history by averaging the Two-Tower item embeddings indexed by the user’s history items; L2-normalize.
    - Compute similarity between user vectors and all item embeddings; take top-K (configurable, e.g., 100).
    - Ensure the true positive is present in the candidate list (for supervised training of the reranker).
    - Save candidate lists for train/val/test with history strings and positive item indices.

    Why: Using history-averaged user vectors provides a content-aware user representation at inference, helping warm-start behavior and capturing recency signals.

    ---

    ### Step 5 — Feature engineering (GPU, sharded)

    Purpose: Build features per (user, candidate) pair that a reranker can exploit.

    Features (for each candidate item c):
    - `dot_uv`: dot product between user vector U (from history) and item vector V[c].
    - `max_sim_recent`: maximum cosine similarity between candidate V[c] and any item vector in the user’s recent history (recency-aware relevance proxy).
    - `pop`: candidate item popularity (normalized frequency).
    - `hist_len`: normalized history length in the window.
    - `price_z`: standardized median price per item (if available from `items_clean`).
    - `label`: 1 if candidate is the true positive, else 0.

    Implementation details:
    - Batches of candidates are processed on GPU.
    - Sharded Parquet files are written for large-scale training (e.g., `ranker_feats_train_shards/part_*.parquet`).

    Why: Separating feature construction from training scales to large datasets and allows offline feature enrichment.

    ---

    ### Step 6 — Reranker model and training

    Model: `RankerMLP`
    - Inputs: concatenated feature vector `[dot_uv, max_sim_recent, pop, hist_len, price_z]`.
    - Architecture: wider MLP with residual skip: `FC→ReLU→Dropout→LayerNorm → FC→ReLU→Dropout→LayerNorm → residual add → FC→ReLU→Dropout → Out(1)`.
    - Regularization: feature-drop on `dot_uv` with probability p during training (reduces overreliance on the strongest feature and improves generalization).

    Training:
    - Loss: `BCEWithLogitsLoss` on labels (1 for positive candidate, 0 for negatives), optionally with hard-negative subsampling per query.
    - Batching: mini-batches sampled from feature shards with random permutations.
    - Early stopping: monitor a dev slice (derived from train by time) or use validation directly; save best model checkpoint (`ranker_best.pt`).

    Evaluation:
    - Within-candidate metrics: For each query, re-score the candidate set and measure Recall@K and NDCG@K by checking whether the positive candidate is ranked within top-K.
    - End-to-end metrics (recommended): Chain retrieval and rerank to compute Recall/NDCG@K over the full catalog. This is not forced during training but should be reported for business evaluation.

    Why: The reranker uses richer, non-linear combinations of features to improve ordering of a compact candidate set, which increases precision at the top-K shown to users.

    ---

    ### Metrics and evaluation protocol

    - Splits are chronological (train/val/test by timestamps) to reflect future prediction.
    - For baselines and Two-Tower validation, seen items are masked per user; the held-out positive is allowed in the candidate set for fair evaluation.
    - Reranker metrics reported in the notebook are within-candidate. For business-facing evaluation, report end-to-end metrics that multiply retrieval coverage by rerank quality.

    Key metrics:
    - Recall@K: fraction of cases where the target is in the top-K.
    - NDCG@K: rank-sensitive variant rewarding higher placement.
    - MRR@K: reciprocal rank, sensitive to the first relevant item.
    - Recommended business KPIs: revenue-weighted NDCG, coverage, novelty/diversity.

    ---

    ### Configuration and reproducibility

    - Seeds set globally (Python, NumPy, PyTorch, CUDA) for repeatability.
    - All major hyperparameters are grouped into `CFG` dicts in notebooks 03 and 04, including dimensions, batch sizes, learning rates, top-K sizes, and sharding.

    ---

    ### How to run

    1) Data preparation
    - Open `notebooks/01_data_prep.ipynb` and run all cells.
    - Verify `interactions_clean.parquet`, ID maps, and sequences are saved.

    2) Baselines
    - Open `notebooks/02_embeddings_and_baselines.ipynb` and run all cells.
    - Confirm `baseline_results.json` is written.

    3) Two-Tower retriever
    - Open `notebooks/03_retriever_and_index.ipynb` and run all cells.
    - Confirm `user_embeddings.npy` and `item_embeddings.npy` are saved.

    4) Reranker
    - Open `notebooks/04_ranker_and_eval.ipynb` and run all cells.
    - Confirm candidates and sharded features are produced; `ranker_best.pt` and `ranker_results.json` are saved.

    ---

    ### Extensibility and production notes

    - Retrieval ANN: load `item_embeddings.npy` into a FAISS IVF/HNSW index for sub-ms nearest-neighbor queries; produce top-100 candidates; rerank with the MLP.
    - Cold start: use popularity+brand/category priors; for new items, rely on metadata features (price, brand, category) and nearest neighbors.
    - Context features: add showroom, sales channel, seasonality (month/quarter), and recency buckets to user/item towers and/or ranker features.
    - Monitoring: track Recall/NDCG@K, coverage, diversity, and revenue uplift over time; implement offline→online drift checks for embeddings and features.

    ---

    ### Summary

    - The system transforms raw transactions into supervised sequences, establishes baselines, learns retrieval embeddings with a Two-Tower model, generates candidates, builds efficient GPU features, and trains an MLP reranker to maximize top-K precision.
    - The architecture is standard, scalable, and production-friendly; the main reporting addition recommended is chained end-to-end metrics and revenue-weighted KPIs for business decisions.

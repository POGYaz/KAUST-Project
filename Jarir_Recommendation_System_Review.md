### Jarir B2B Recommendation System — Comprehensive Code & Logic Review

This document reviews the full Two-Tower + Reranker pipeline implemented in notebooks 01–04, plus relevant source modules. It covers correctness of data processing, model logic, evaluation, and the validity of reported results, and finishes with prioritized, actionable recommendations.

---

### Executive summary

- The end-to-end design is solid: a clean preprocessing pipeline, a Two-Tower retriever, and a learned MLP reranker over TT-based candidates.
- Baselines are correctly implemented and evaluated with temporal holdout; ItemKNN is the strongest baseline.
- The Two-Tower retriever improves over the best baseline at Recall@10 on validation.
- The reranker achieves strong within-candidate ranking metrics (e.g., Recall@10 ≈ 0.73 on a dev slice constructed from train), but this is not yet the end-to-end metric over the full item catalog.
- No critical leakage found in current code paths; small risks remain around negative sampling and metric interpretation.

Overall: the logic is correct, the implementation quality is good, and the results are promising for B2B Jarir. The main gap is to report end-to-end metrics that chain retrieval+rerank, alongside business KPIs.

---

### 01 — Data preparation (`notebooks/01_data_prep.ipynb`)

**What it does well**
- Robust date parsing for Jarir-specific formats and Excel serials; raises an error if all dates fail to parse (no synthetic timestamps).
- Clean filters: missing-core fields, returns (by prefix), positive price/quantity, duplicates removal.
- Log-scale IQR-based outlier handling with configurable winsorize/drop.
- Text normalization and non-product line filtering.
- Temporal sequence building with global 80/10/10 time split and minimum history.
- Saves interaction tables and ID maps; reports basic dataset stats.

**Checks passed**
- Price calculation guards against division-by-zero and non-finite values.
- All splits are time-based using per-row timestamps.

**Notes / minor risks**
- Outlier thresholds are global; consider per-category or robust-by-brand options later.
- Coverage filters are currently MIN=1; increase if you want stricter cold-start handling.

---

### 02 — Embeddings & baselines (`notebooks/02_embeddings_and_baselines.ipynb`)

**What it does**
- Builds a user–item matrix from sequences (history weighted lower than the positive), with an additional denser KNN source matrix constructed from interactions up to the train cutoff.
- Implements Popularity, ItemKNN (cosine item–item), UserKNN, and NMF baselines.
- Evaluates with Recall@K, NDCG@K, and MRR@K on held-out validation interactions, masking seen items but allowing the held-out positive.

**Observed metrics (Validation, K=10)**
- Popularity: Recall ~0.0947
- ItemKNN: Recall ~0.1124 (best baseline)
- UserKNN: Recall ~0.0118
- Matrix Factorization (NMF): Recall ~0.0059

These are consistent with sparse retail data where item co-occurrence beats popularity modestly, and user-user is weak due to low per-user density.

**Checks passed / risks**
- Temporal holdout is respected by zeroing validation positives in the train matrix and by cutting KNN source interactions at the train cutoff.
- ItemKNN is implemented correctly (cosine item–item; user profile from the same base matrix) and no masking bug is present.

---

### 03 — Two-Tower retriever (`notebooks/03_retriever_and_index.ipynb`)

**Architecture & training**
- Two towers with learned `Embedding` layers for users and items (dimension 256), residual blocks with LayerNorm and Dropout, and L2-normalized outputs for cosine scoring.
- InfoNCE-style cross-entropy with temperature; per-example random negatives sampled from the catalog excluding history and the positive.
- Time-based dev split from train for early stopping and LR scheduling; validation used only for the final comparison.

**Observed metrics (Validation, K=10)**
- Two-Tower Recall@10 ≈ 0.1243 vs best baseline ≈ 0.1124.

This is a plausible, incremental gain for this data size (n_items ≈ 1,735). More gains typically come from richer features (contexts, item text, price, recency) and better negatives.

**Notes / risks**
- Negative sampling may include future-bought items for a user (not present in the per-example history). This is common and usually acceptable; you can mitigate with popularity-aware or in-batch hard negatives.
- User tower uses only `user_id`. In the reranker stage you construct user vectors from histories (averaging item vectors). The difference is intentional but worth documenting to avoid confusion.

---

### 04 — Two-stage reranker (`notebooks/04_ranker_and_eval.ipynb`)

**Candidate generation**
- Uses Two-Tower item embeddings; constructs user vectors from history by averaging the item vectors; retrieves top-K by dot-product; ensures the positive is present in the candidate set.

**Feature engineering**
- Features per (user, candidate): dot(u,v), max-sim-to-recent-history, popularity, normalized hist length, optional price z-score. All computed efficiently on GPU and saved as sharded Parquet.

**Ranker**
- MLP with residual skip, LayerNorms, dropout, and feature-drop regularization on dot(u,v) to reduce overreliance. Trained with BCE-with-logits; early stopping on a dev slice derived from train.

**Observed metrics**
- Dev Recall@10 improves steadily and reaches ≈ 0.7257. Final validation/test metrics are computed after training, but the exact numbers weren’t persisted in versioned JSON in this snapshot. The magnitude (≈0.7 within-candidate Recall@10) is believable for 100-size candidate sets.

Important: these reranker metrics are within-candidate metrics (conditional on the positive being present among K candidates). They are not the end-to-end Recall@10 over the full item catalog.

---

### Are the results “right and good”?

- Baseline metrics are sensible for the dataset scale and sparsity.
- The Two-Tower retriever improves over the strongest baseline at Recall@10, which is the expected behavior and indicates correct training/evaluation.
- The reranker shows strong within-candidate ranking, which is good; however, the business-facing end-to-end Recall@K should combine retrieval and ranking: end_to_end@K = P(retriever puts positive in candidate set) × P(reranker ranks positive into top-K | in set).
- Because candidates intentionally include the positive, reranker metrics don’t answer end-to-end performance yet. They are still very useful for comparing rankers.

Conclusion: the code and logic are correct; results are promising and credible. To answer “is the result right and good?” for business decisions, report end-to-end metrics and revenue-weighted KPIs in addition to within-candidate metrics.

---

### Gaps, risks, and how to close them

- End-to-end metrics missing: compute Recall/NDCG/MRR@{10,20,100} over the full catalog by chaining retrieval and rerank.
- Candidate generation mismatch: retriever user vectors are learned from `user_id`, but reranker candidate generation uses history-averaged item vectors. This is okay, but document the design or align both stages.
- Negative sampling: consider popularity-aware or semi-hard negatives; optionally exclude items purchased by the same user later in time if leakage is a concern.
- Small sample sizes: variance can be high; add confidence intervals or time-based backtests.
- Dependencies: torch==2.6.0 and `faiss-gpu-cu12` may be brittle on Windows; simplify for local CPU workflows or pin CUDA versions explicitly.

---

### Actionable recommendations (prioritized)

1) Reporting and evaluation
- Compute and log end-to-end metrics (retriever@{10,50,100}, reranker-within-candidate@{5,10}, and chained end_to_end@{5,10}).
- Add per-segment breakdowns (by customer size/category, top brands, price bands).
- Add revenue-weighted Recall/NDCG and Coverage/Novelty/Diversity.

2) Retriever improvements
- Try in-batch hard negatives or mined negatives (e.g., ItemKNN top-N excluding positives) to accelerate learning.
- Add simple contextual features: recency buckets, country/showroom, sales channel, and condition them via learned embeddings.
- Optionally derive user vectors from recent history at inference (hybrid of `user_id` and history pooling) for cold/warm start.

3) Reranker improvements
- Calibrate scores (Platt/Isotonic) to improve ranking stability across users.
- Add cross-features (e.g., price gap vs user median, brand affinity, category novelty).
- Evaluate pairwise/ranking losses (BPR, LambdaRank) vs BCE.

4) Productionization
- Build a FAISS HNSW/IVF index over item embeddings; measure recall/speed tradeoffs.
- Create simple Python serving for candidate gen + rerank; add guardrails/fallbacks to popularity when embeddings are stale.
- Add unit tests for sequence building, masking, and metrics; persist seeds/configs with runs.

---

### Quick facts (from current run logs)

- Data scale: ~1,735 items; ~929 users; sequences: train 1,108 / val 169 / test 160.
- Best baseline (K=10, validation): ItemKNN Recall ≈ 0.1124.
- Two-Tower (K=10, validation): Recall ≈ 0.1243.
- Reranker (within-candidate, dev slice, K=10): Recall ≈ 0.7257.

---

### Final verdict

The implementation is correct and reasonably optimized. Baseline and Two-Tower numbers are aligned with expectations. The reranker achieves strong within-candidate ranking quality. For decision-grade reporting, add chained end-to-end metrics and business KPIs; with those in place, this system is in good shape for pilot deployment.

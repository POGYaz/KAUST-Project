"""
Evaluation pipeline for the recommendation system.

This module provides comprehensive evaluation pipelines for both
retrieval and ranking models, including baseline comparisons and
performance analysis.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..data.dataset import create_interaction_matrix, parse_history_string
from ..utils.io import read_json, read_numpy, read_parquet, write_json
from ..utils.logging import get_logger
from .metrics import compute_all_metrics, coverage_at_k

logger = get_logger(__name__)


class BaselineEvaluator:
    """
    Evaluator for baseline recommendation models.
    
    Implements and evaluates standard baseline methods including
    popularity, item-based KNN, user-based KNN, and matrix factorization.
    """
    
    def __init__(
        self,
        k_values: List[int] = [5, 10, 20],
        knn_neighbors: int = 50,
        mf_components: int = 50,
        pop_alpha: float = 0.01,
        random_seed: int = 42,
    ):
        """
        Initialize the baseline evaluator.
        
        Args:
            k_values: List of K values for evaluation.
            knn_neighbors: Number of neighbors for KNN models.
            mf_components: Number of components for matrix factorization.
            pop_alpha: Popularity regularization parameter.
            random_seed: Random seed for reproducibility.
        """
        self.k_values = k_values
        self.knn_neighbors = knn_neighbors
        self.mf_components = mf_components
        self.pop_alpha = pop_alpha
        self.random_seed = random_seed
        
        # Initialize baseline models
        self.models = {}
    
    def fit_baselines(
        self,
        train_matrix: csr_matrix,
        knn_matrix: Optional[csr_matrix] = None,
    ) -> Dict[str, Any]:
        """
        Fit all baseline models on training data.
        
        Args:
            train_matrix: Sparse training interaction matrix.
            knn_matrix: Optional denser matrix for KNN similarity computation.
            
        Returns:
            Dictionary of fitted baseline models.
        """
        logger.info("Fitting baseline models")
        
        if knn_matrix is None:
            knn_matrix = train_matrix
        
        # Popularity baseline
        self.models['Popularity'] = PopularityModel()
        self.models['Popularity'].fit(train_matrix)
        
        # Item-based KNN
        self.models['ItemKNN'] = ItemKNNModel(
            k=self.knn_neighbors,
            pop_alpha=self.pop_alpha
        )
        self.models['ItemKNN'].fit(knn_matrix)
        
        # User-based KNN
        self.models['UserKNN'] = UserKNNModel(k=self.knn_neighbors)
        self.models['UserKNN'].fit(train_matrix)
        
        # Matrix Factorization
        self.models['MatrixFactorization'] = MatrixFactorizationModel(
            n_components=self.mf_components,
            random_seed=self.random_seed
        )
        self.models['MatrixFactorization'].fit(train_matrix)
        
        logger.info(f"Fitted {len(self.models)} baseline models")
        return self.models
    
    def evaluate_baselines(
        self,
        test_interactions: List[Tuple[int, int]],
        train_matrix: csr_matrix,
        total_items: int,
    ) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Evaluate all baseline models on test interactions.
        
        Args:
            test_interactions: List of (user_id, true_item_id) tuples.
            train_matrix: Training matrix for filtering seen items.
            total_items: Total number of items for coverage computation.
            
        Returns:
            Nested dictionary with evaluation results.
        """
        logger.info(f"Evaluating baselines on {len(test_interactions)} interactions")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}")
            
            predictions = []
            ground_truth = []
            
            start_time = time.time()
            
            for user_id, true_item_id in test_interactions:
                # Get recommendations
                max_k = max(self.k_values)
                rec_list = model.recommend(
                    user_id, k=max_k, train_matrix=train_matrix
                )
                
                predictions.append(rec_list)
                ground_truth.append(true_item_id)
            
            evaluation_time = time.time() - start_time
            
            # Compute metrics
            model_metrics = compute_all_metrics(
                predictions, ground_truth, self.k_values, total_items
            )
            
            # Add timing information
            model_metrics['evaluation_time'] = evaluation_time
            model_metrics['avg_time_per_query'] = evaluation_time / len(test_interactions)
            
            results[model_name] = model_metrics
            
            # Log results
            for k in self.k_values:
                recall = model_metrics['Recall'][k]
                ndcg = model_metrics['NDCG'][k]
                logger.info(f"{model_name} - Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")
        
        return results


class PopularityModel:
    """Simple popularity-based recommendation model."""
    
    def __init__(self):
        self.popularity_scores = None
    
    def fit(self, train_matrix: csr_matrix) -> 'PopularityModel':
        """Fit the popularity model."""
        self.popularity_scores = np.array(train_matrix.sum(axis=0)).flatten()
        return self
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> List[int]:
        """Generate recommendations based on popularity."""
        scores = self.popularity_scores.copy()
        
        # Filter out items the user has already interacted with
        if train_matrix is not None:
            seen_items = train_matrix[user_id].nonzero()[1]
            scores[seen_items] = -1e12
        
        # Return top-k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices.tolist()


class ItemKNNModel:
    """Item-based K-nearest neighbors recommendation model."""
    
    def __init__(self, k: int = 50, pop_alpha: float = 0.01):
        self.k = k
        self.pop_alpha = pop_alpha
        self.similarity_matrix = None
        self.popularity_scores = None
        self.base_matrix = None
    
    def fit(self, train_matrix: csr_matrix) -> 'ItemKNNModel':
        """Fit the item KNN model."""
        self.base_matrix = train_matrix
        
        # Compute item-item cosine similarity
        self.similarity_matrix = cosine_similarity(train_matrix.T)
        
        # Compute popularity scores for regularization
        self.popularity_scores = np.array(train_matrix.sum(axis=0)).flatten().astype(np.float32)
        if self.popularity_scores.max() > 0:
            self.popularity_scores /= self.popularity_scores.max()
        
        return self
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> List[int]:
        """Generate recommendations using item-based collaborative filtering."""
        # Get user's interaction vector
        user_vector = self.base_matrix[user_id].toarray().flatten()
        
        # Compute recommendation scores
        scores = user_vector @ self.similarity_matrix
        
        # Add popularity regularization
        scores = scores + self.pop_alpha * self.popularity_scores
        
        # Filter out items the user has already interacted with
        seen_items = np.where(user_vector > 0)[0]
        scores[seen_items] = -1e12
        
        # Return top-k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices.tolist()


class UserKNNModel:
    """User-based K-nearest neighbors recommendation model."""
    
    def __init__(self, k: int = 50):
        self.k = k
        self.similarity_matrix = None
        self.train_matrix = None
    
    def fit(self, train_matrix: csr_matrix) -> 'UserKNNModel':
        """Fit the user KNN model."""
        self.train_matrix = train_matrix
        
        # Compute user-user cosine similarity
        self.similarity_matrix = cosine_similarity(train_matrix)
        
        return self
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> List[int]:
        """Generate recommendations using user-based collaborative filtering."""
        # Find similar users (excluding the user themselves)
        user_similarities = self.similarity_matrix[user_id].copy()
        user_similarities[user_id] = -1e12
        
        # Get top-k similar users
        similar_users = np.argsort(user_similarities)[-self.k:][::-1]
        
        # Compute recommendation scores
        scores = np.zeros(self.train_matrix.shape[1])
        for similar_user in similar_users:
            similarity_weight = max(user_similarities[similar_user], 0)
            user_items = self.train_matrix[similar_user].toarray().flatten()
            scores += user_items * similarity_weight
        
        # Filter out items the user has already interacted with
        if train_matrix is not None:
            seen_items = train_matrix[user_id].nonzero()[1]
        else:
            seen_items = self.train_matrix[user_id].nonzero()[1]
        scores[seen_items] = -1e12
        
        # Return top-k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices.tolist()


class MatrixFactorizationModel:
    """Matrix factorization recommendation model using NMF."""
    
    def __init__(self, n_components: int = 50, random_seed: int = 42):
        self.n_components = n_components
        self.random_seed = random_seed
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None
    
    def fit(self, train_matrix: csr_matrix) -> 'MatrixFactorizationModel':
        """Fit the matrix factorization model."""
        from sklearn.decomposition import NMF
        
        self.train_matrix = train_matrix
        
        # Convert to dense for NMF
        dense_matrix = train_matrix.toarray()
        
        # Fit NMF model
        nmf_model = NMF(
            n_components=self.n_components,
            random_state=self.random_seed
        )
        
        self.user_factors = nmf_model.fit_transform(dense_matrix)
        self.item_factors = nmf_model.components_
        
        return self
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> List[int]:
        """Generate recommendations using matrix factorization."""
        # Compute recommendation scores
        user_vector = self.user_factors[user_id]
        scores = user_vector.dot(self.item_factors)
        
        # Filter out items the user has already interacted with
        if train_matrix is not None:
            seen_items = train_matrix[user_id].nonzero()[1]
        else:
            seen_items = self.train_matrix[user_id].nonzero()[1]
        scores[seen_items] = -1e12
        
        # Return top-k items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices.tolist()


class RetrievalEvaluator:
    """
    Evaluator for neural retrieval models.
    
    Evaluates Two-Tower and other embedding-based retrieval models
    by computing user and item embeddings and measuring retrieval performance.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50, 100]):
        """
        Initialize the retrieval evaluator.
        
        Args:
            k_values: List of K values for evaluation.
        """
        self.k_values = k_values
    
    def evaluate_retrieval(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        test_interactions: List[Tuple[int, int]],
        total_items: int,
        batch_size: int = 1000,
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate retrieval performance using embeddings.
        
        Args:
            user_embeddings: User embedding matrix.
            item_embeddings: Item embedding matrix.
            test_interactions: List of (user_id, true_item_id) tuples.
            total_items: Total number of items for coverage computation.
            batch_size: Batch size for efficient computation.
            
        Returns:
            Dictionary with retrieval metrics.
        """
        logger.info(f"Evaluating retrieval on {len(test_interactions)} interactions")
        
        predictions = []
        ground_truth = []
        
        # Process interactions in batches for memory efficiency
        for i in range(0, len(test_interactions), batch_size):
            batch_interactions = test_interactions[i:i + batch_size]
            
            batch_users = [user_id for user_id, _ in batch_interactions]
            batch_true_items = [item_id for _, item_id in batch_interactions]
            
            # Get user embeddings for this batch
            batch_user_emb = user_embeddings[batch_users]
            
            # Compute similarities with all items
            similarities = batch_user_emb @ item_embeddings.T
            
            # Get top-k recommendations for each user
            max_k = max(self.k_values)
            top_k_indices = np.argsort(similarities, axis=1)[:, -max_k:][:, ::-1]
            
            # Store predictions and ground truth
            for j, (user_id, true_item_id) in enumerate(batch_interactions):
                predictions.append(top_k_indices[j].tolist())
                ground_truth.append(true_item_id)
        
        # Compute metrics
        metrics = compute_all_metrics(
            predictions, ground_truth, self.k_values, total_items
        )
        
        return metrics


class RankingEvaluator:
    """
    Evaluator for ranking models.
    
    Evaluates MLP rerankers and other ranking models by reranking
    candidate lists and measuring ranking performance.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize the ranking evaluator.
        
        Args:
            k_values: List of K values for evaluation.
        """
        self.k_values = k_values
    
    def evaluate_ranking(
        self,
        model: Any,
        candidates_df: pd.DataFrame,
        feature_builder: Any,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        total_items: int,
        batch_size: int = 1024,
        device: str = 'cpu',
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate ranking performance using a trained ranker.
        
        Args:
            model: Trained ranking model.
            candidates_df: DataFrame with candidate lists.
            feature_builder: Feature builder for creating ranking features.
            user_embeddings: User embedding matrix.
            item_embeddings: Item embedding matrix.
            total_items: Total number of items for coverage computation.
            batch_size: Batch size for processing.
            device: Device for tensor computations.
            
        Returns:
            Dictionary with ranking metrics.
        """
        logger.info(f"Evaluating ranking on {len(candidates_df)} queries")
        
        model.eval()
        predictions = []
        ground_truth = []
        
        # Process candidates in batches
        for i in range(0, len(candidates_df), batch_size):
            batch_df = candidates_df.iloc[i:i + batch_size]
            
            # Rerank candidates for this batch
            batch_predictions = self._rerank_batch(
                model, batch_df, feature_builder,
                user_embeddings, item_embeddings, device
            )
            
            predictions.extend(batch_predictions)
            ground_truth.extend(batch_df['pos_item_idx'].tolist())
        
        # Compute metrics
        metrics = compute_all_metrics(
            predictions, ground_truth, self.k_values, total_items
        )
        
        return metrics
    
    def _rerank_batch(
        self,
        model: Any,
        batch_df: pd.DataFrame,
        feature_builder: Any,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        device: str,
    ) -> List[List[int]]:
        """Rerank candidates for a batch of queries."""
        import torch
        
        # Build features for the batch
        batch_features = feature_builder.build_features(
            batch_df, user_embeddings, item_embeddings, device=device
        )
        
        # Convert features to tensor
        feature_names = ['dot_uv', 'max_sim_recent', 'pop', 'hist_len', 'price_z']
        X = torch.stack([
            torch.from_numpy(batch_features[name]).to(device)
            for name in feature_names
        ], dim=-1)
        
        # Get ranking scores
        with torch.no_grad():
            scores = model(X.view(-1, len(feature_names)))
        
        # Reshape scores and get top-k
        batch_size = len(batch_df)
        candidates_per_query = X.size(1)
        scores = scores.view(batch_size, candidates_per_query)
        
        max_k = max(self.k_values)
        _, top_k_indices = torch.topk(scores, k=min(max_k, candidates_per_query), dim=1)
        
        # Convert to item indices
        predictions = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            candidate_items = [int(x) for x in row['cands'].split()]
            reranked_items = [candidate_items[idx] for idx in top_k_indices[i].cpu().tolist()]
            predictions.append(reranked_items)
        
        return predictions


def run_full_evaluation(
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of all models.
    
    Args:
        data_dir: Directory containing processed data.
        model_dir: Directory containing trained models.
        output_dir: Directory to save evaluation results.
        k_values: List of K values for evaluation.
        
    Returns:
        Dictionary with all evaluation results.
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Running full evaluation pipeline")
    
    # Load data
    sequences = {}
    for split in ['train', 'val', 'test']:
        sequences[split] = read_parquet(data_dir / f'sequences_{split}.parquet')
    
    item_map, customer_map = load_id_mappings(data_dir)
    
    # Create interaction matrices
    n_users, n_items = len(customer_map), len(item_map)
    train_matrix = create_interaction_matrix(sequences['train'], n_users, n_items)
    
    # Prepare test interactions
    test_interactions = [
        (int(row['user_idx']), int(row['pos_item_idx']))
        for _, row in sequences['test'].iterrows()
    ]
    
    results = {}
    
    # Evaluate baselines
    baseline_evaluator = BaselineEvaluator(k_values=k_values)
    baseline_models = baseline_evaluator.fit_baselines(train_matrix)
    baseline_results = baseline_evaluator.evaluate_baselines(
        test_interactions, train_matrix, n_items
    )
    results['baselines'] = baseline_results
    
    # Evaluate retrieval model if available
    user_emb_path = model_dir / 'user_embeddings.npy'
    item_emb_path = model_dir / 'item_embeddings.npy'
    
    if user_emb_path.exists() and item_emb_path.exists():
        user_embeddings = read_numpy(user_emb_path)
        item_embeddings = read_numpy(item_emb_path)
        
        retrieval_evaluator = RetrievalEvaluator(k_values=k_values)
        retrieval_results = retrieval_evaluator.evaluate_retrieval(
            user_embeddings, item_embeddings, test_interactions, n_items
        )
        results['retrieval'] = retrieval_results
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    write_json(results, results_path)
    
    logger.info(f"Evaluation complete. Results saved to {results_path}")
    return results

"""
Reranking pipeline for refining candidate recommendations.

This module implements the reranking stage of the recommendation pipeline,
using trained MLP rankers to reorder candidates based on rich features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.features import RankingFeatureBuilder
from ..models.reranker.mlp_ranker import MLPRanker
from ..utils.io import read_numpy
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MLPReranker:
    """
    MLP-based reranking system.
    
    Uses a trained MLP ranker to reorder candidate items based on
    engineered features including embeddings, popularity, and metadata.
    """
    
    def __init__(
        self,
        model: MLPRanker,
        feature_builder: RankingFeatureBuilder,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        popularity_scores: Optional[np.ndarray] = None,
        price_features: Optional[np.ndarray] = None,
        device: Union[str, torch.device] = 'cpu',
    ):
        """
        Initialize the MLP reranker.
        
        Args:
            model: Trained MLP ranking model.
            feature_builder: Feature builder for creating ranking features.
            user_embeddings: User embedding matrix.
            item_embeddings: Item embedding matrix.
            popularity_scores: Optional popularity scores for items.
            price_features: Optional price features for items.
            device: Device for model inference.
        """
        self.model = model
        self.feature_builder = feature_builder
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.popularity_scores = popularity_scores
        self.price_features = price_features
        self.device = torch.device(device)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Feature names (should match feature builder)
        self.feature_names = ['dot_uv', 'max_sim_recent', 'pop', 'hist_len', 'price_z']
    
    def rerank_candidates(
        self,
        candidates_df: pd.DataFrame,
        k: int = 10,
        batch_size: int = 1024,
    ) -> pd.DataFrame:
        """
        Rerank candidates for multiple queries.
        
        Args:
            candidates_df: DataFrame with candidate lists.
            k: Number of top candidates to return per query.
            batch_size: Batch size for processing.
            
        Returns:
            DataFrame with reranked candidate lists.
        """
        logger.info(f"Reranking candidates for {len(candidates_df)} queries")
        
        reranked_results = []
        
        # Process in batches
        for i in range(0, len(candidates_df), batch_size):
            batch_df = candidates_df.iloc[i:i + batch_size]
            batch_results = self._rerank_batch(batch_df, k)
            reranked_results.extend(batch_results)
        
        # Create result DataFrame
        result_df = pd.DataFrame(reranked_results)
        logger.info(f"Completed reranking for {len(result_df)} queries")
        
        return result_df
    
    def rerank_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        history_items: List[int],
        k: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank candidates for a single user.
        
        Args:
            user_id: User ID.
            candidate_items: List of candidate item IDs.
            history_items: List of user's history item IDs.
            k: Number of top candidates to return.
            
        Returns:
            Tuple of (reranked_item_ids, scores).
        """
        # Create a temporary DataFrame for this query
        query_df = pd.DataFrame([{
            'user_idx': user_id,
            'history_idx': ' '.join(map(str, history_items)),
            'cands': ' '.join(map(str, candidate_items)),
            'pos_item_idx': candidate_items[0],  # Dummy positive
        }])
        
        # Rerank
        result_df = self._rerank_batch(query_df, k)
        
        if result_df:
            reranked_items = [int(x) for x in result_df[0]['reranked_cands'].split()]
            scores = result_df[0].get('scores', [1.0] * len(reranked_items))
            return reranked_items, scores
        else:
            return candidate_items[:k], [1.0] * min(k, len(candidate_items))
    
    def _rerank_batch(
        self,
        batch_df: pd.DataFrame,
        k: int,
    ) -> List[Dict]:
        """Rerank candidates for a batch of queries."""
        batch_results = []
        
        # Build features for the batch
        features = self._build_features_for_batch(batch_df)
        
        if not features:
            # Fallback: return original candidates
            for _, row in batch_df.iterrows():
                candidates = [int(x) for x in row['cands'].split()]
                batch_results.append({
                    'history_idx': str(row.get('history_idx', '')),
                    'pos_item_idx': int(row['pos_item_idx']),
                    'reranked_cands': ' '.join(map(str, candidates[:k])),
                    'scores': [1.0] * min(k, len(candidates)),
                })
            return batch_results
        
        # Convert features to tensor
        feature_tensor = self._features_to_tensor(features)
        
        # Get ranking scores
        with torch.no_grad():
            scores = self.model.predict_scores(feature_tensor)
        
        # Reshape scores and rerank
        batch_size = len(batch_df)
        candidates_per_query = feature_tensor.size(0) // batch_size
        scores = scores.view(batch_size, candidates_per_query)
        
        # Process each query in the batch
        for i, (_, row) in enumerate(batch_df.iterrows()):
            candidates = [int(x) for x in row['cands'].split()]
            query_scores = scores[i].cpu().numpy()
            
            # Get top-k indices
            top_k = min(k, len(candidates))
            top_k_indices = np.argsort(query_scores)[-top_k:][::-1]
            
            # Rerank candidates
            reranked_candidates = [candidates[idx] for idx in top_k_indices]
            reranked_scores = [float(query_scores[idx]) for idx in top_k_indices]
            
            batch_results.append({
                'history_idx': str(row.get('history_idx', '')),
                'pos_item_idx': int(row['pos_item_idx']),
                'reranked_cands': ' '.join(map(str, reranked_candidates)),
                'scores': reranked_scores,
            })
        
        return batch_results
    
    def _build_features_for_batch(self, batch_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Build ranking features for a batch of queries."""
        try:
            features = self.feature_builder.build_features(
                batch_df,
                self.user_embeddings,
                self.item_embeddings,
                self.popularity_scores,
                self.price_features,
                device=str(self.device),
                batch_size=len(batch_df),
            )
            return features
        except Exception as e:
            logger.warning(f"Feature building failed: {e}")
            return {}
    
    def _features_to_tensor(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert feature dictionary to tensor."""
        # Stack features in the expected order
        feature_arrays = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_arrays.append(features[feature_name])
            else:
                # Create zero features as fallback
                shape = (features[self.feature_names[0]].shape[0],) if features else (1,)
                feature_arrays.append(np.zeros(shape, dtype=np.float32))
        
        # Stack along feature dimension
        feature_matrix = np.stack(feature_arrays, axis=-1)
        
        # Convert to tensor
        return torch.from_numpy(feature_matrix).to(self.device)


class EnsembleReranker:
    """
    Ensemble reranker combining multiple ranking models.
    
    Combines predictions from multiple rankers using weighted averaging
    or other ensemble methods.
    """
    
    def __init__(
        self,
        rankers: Dict[str, MLPReranker],
        weights: Optional[Dict[str, float]] = None,
        ensemble_method: str = 'average',
    ):
        """
        Initialize ensemble reranker.
        
        Args:
            rankers: Dictionary of ranker name to ranker instance.
            weights: Optional weights for combining rankers.
            ensemble_method: Method for combining predictions ('average', 'max', 'min').
        """
        self.rankers = rankers
        self.weights = weights or {name: 1.0 for name in rankers.keys()}
        self.ensemble_method = ensemble_method
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: w / total_weight for name, w in self.weights.items()}
    
    def rerank_candidates(
        self,
        candidates_df: pd.DataFrame,
        k: int = 10,
        batch_size: int = 1024,
    ) -> pd.DataFrame:
        """
        Rerank candidates using ensemble approach.
        
        Args:
            candidates_df: DataFrame with candidate lists.
            k: Number of top candidates to return per query.
            batch_size: Batch size for processing.
            
        Returns:
            DataFrame with ensemble reranked candidates.
        """
        logger.info(f"Ensemble reranking for {len(candidates_df)} queries")
        
        # Get predictions from all rankers
        all_predictions = {}
        
        for name, ranker in self.rankers.items():
            try:
                predictions = ranker.rerank_candidates(candidates_df, k=k, batch_size=batch_size)
                all_predictions[name] = predictions
            except Exception as e:
                logger.warning(f"Ranker {name} failed: {e}")
                continue
        
        if not all_predictions:
            logger.error("No rankers produced valid predictions")
            return candidates_df
        
        # Combine predictions
        return self._combine_predictions(all_predictions, candidates_df, k)
    
    def _combine_predictions(
        self,
        all_predictions: Dict[str, pd.DataFrame],
        original_df: pd.DataFrame,
        k: int,
    ) -> pd.DataFrame:
        """Combine predictions from multiple rankers."""
        combined_results = []
        
        for i, (_, row) in enumerate(original_df.iterrows()):
            # Collect scores for each candidate from all rankers
            candidates = [int(x) for x in row['cands'].split()]
            candidate_scores = {item_id: [] for item_id in candidates}
            
            # Extract scores from each ranker
            for name, predictions in all_predictions.items():
                weight = self.weights[name]
                
                if i < len(predictions):
                    pred_row = predictions.iloc[i]
                    reranked_items = [int(x) for x in pred_row['reranked_cands'].split()]
                    scores = pred_row.get('scores', [1.0] * len(reranked_items))
                    
                    # Map scores to candidates
                    for item_id, score in zip(reranked_items, scores):
                        if item_id in candidate_scores:
                            candidate_scores[item_id].append(weight * score)
            
            # Combine scores for each candidate
            combined_scores = {}
            for item_id, score_list in candidate_scores.items():
                if score_list:
                    if self.ensemble_method == 'average':
                        combined_scores[item_id] = np.mean(score_list)
                    elif self.ensemble_method == 'max':
                        combined_scores[item_id] = np.max(score_list)
                    elif self.ensemble_method == 'min':
                        combined_scores[item_id] = np.min(score_list)
                    else:
                        combined_scores[item_id] = np.mean(score_list)
                else:
                    combined_scores[item_id] = 0.0
            
            # Sort by combined score
            sorted_candidates = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            reranked_items = [item_id for item_id, _ in sorted_candidates]
            reranked_scores = [score for _, score in sorted_candidates]
            
            combined_results.append({
                'history_idx': str(row.get('history_idx', '')),
                'pos_item_idx': int(row['pos_item_idx']),
                'reranked_cands': ' '.join(map(str, reranked_items)),
                'scores': reranked_scores,
            })
        
        return pd.DataFrame(combined_results)


def load_reranker(
    model_path: Union[str, Path],
    user_embeddings_path: Union[str, Path],
    item_embeddings_path: Union[str, Path],
    popularity_scores_path: Optional[Union[str, Path]] = None,
    price_features_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = 'cpu',
) -> MLPReranker:
    """
    Load a trained MLP reranker from disk.
    
    Args:
        model_path: Path to saved model state dict.
        user_embeddings_path: Path to user embeddings.
        item_embeddings_path: Path to item embeddings.
        popularity_scores_path: Optional path to popularity scores.
        price_features_path: Optional path to price features.
        device: Device for model inference.
        
    Returns:
        Loaded MLPReranker instance.
    """
    # Load model (this would need model architecture info)
    # For now, assume model is loaded elsewhere
    model = None  # This needs to be implemented based on saved model format
    
    # Load embeddings
    user_embeddings = read_numpy(user_embeddings_path)
    item_embeddings = read_numpy(item_embeddings_path)
    
    # Load optional features
    popularity_scores = None
    if popularity_scores_path is not None:
        popularity_scores = read_numpy(popularity_scores_path)
    
    price_features = None
    if price_features_path is not None:
        price_features = read_numpy(price_features_path)
    
    # Create feature builder
    feature_builder = RankingFeatureBuilder(
        embedding_dim=item_embeddings.shape[1],
        max_history_length=15,  # Default value
    )
    
    # Create reranker
    reranker = MLPReranker(
        model=model,
        feature_builder=feature_builder,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        popularity_scores=popularity_scores,
        price_features=price_features,
        device=device,
    )
    
    return reranker


def evaluate_reranking(
    reranker: MLPReranker,
    test_candidates: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate reranking performance.
    
    Args:
        reranker: Reranker to evaluate.
        test_candidates: Test candidates DataFrame.
        k_values: List of K values to evaluate.
        
    Returns:
        Dictionary with reranking metrics.
    """
    logger.info(f"Evaluating reranking on {len(test_candidates)} queries")
    
    metrics = {}
    
    for k in k_values:
        # Rerank candidates
        reranked_df = reranker.rerank_candidates(test_candidates, k=k)
        
        hits = 0
        total = 0
        ndcg_sum = 0.0
        
        for _, row in reranked_df.iterrows():
            positive_item = int(row['pos_item_idx'])
            reranked_items = [int(x) for x in row['reranked_cands'].split()]
            
            # Check if positive item is in top-k
            if positive_item in reranked_items:
                hits += 1
                
                # Compute NDCG contribution
                rank = reranked_items.index(positive_item) + 1
                ndcg_sum += 1.0 / np.log2(rank + 1)
            
            total += 1
        
        # Compute metrics
        recall = hits / max(total, 1)
        ndcg = ndcg_sum / max(total, 1)
        
        metrics[f'recall@{k}'] = recall
        metrics[f'ndcg@{k}'] = ndcg
        
        logger.info(f"Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")
    
    return metrics

"""
Retrieval pipeline for generating candidate recommendations.

This module implements the retrieval stage of the recommendation pipeline,
using trained Two-Tower models and ANN indices to efficiently generate
candidate items for users.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.dataset import parse_history_string
from ..models.retriever.two_tower import TwoTowerModel
from ..utils.io import read_numpy, read_parquet
from ..utils.logging import get_logger
from .ann_index import ANNIndex

logger = get_logger(__name__)


class TwoTowerRetriever:
    """
    Two-Tower based retrieval system.
    
    Uses a trained Two-Tower model to generate user embeddings from
    interaction history and retrieve candidate items using an ANN index.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        ann_index: ANNIndex,
        item_embeddings: np.ndarray,
        device: Union[str, torch.device] = 'cpu',
    ):
        """
        Initialize the Two-Tower retriever.
        
        Args:
            model: Trained Two-Tower model.
            ann_index: ANN index for efficient search.
            item_embeddings: Item embedding matrix.
            device: Device for model inference.
        """
        self.model = model
        self.ann_index = ann_index
        self.item_embeddings = item_embeddings
        self.device = torch.device(device)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Validate dimensions
        if item_embeddings.shape[1] != ann_index.dimension:
            raise ValueError(
                f"Item embedding dimension {item_embeddings.shape[1]} "
                f"doesn't match index dimension {ann_index.dimension}"
            )
    
    def retrieve_for_user(
        self,
        user_id: int,
        k: int = 100,
        exclude_items: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k candidate items for a single user.
        
        Args:
            user_id: User ID to generate recommendations for.
            k: Number of candidates to retrieve.
            exclude_items: Optional list of item IDs to exclude.
            
        Returns:
            Tuple of (item_ids, scores).
        """
        # Get user embedding
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.device)
            user_embedding = self.model.user_tower(user_tensor)
            user_embedding = user_embedding.cpu().numpy()
        
        # Search index
        scores, indices = self.ann_index.search(user_embedding, k=k)
        
        # Extract results for single user
        item_ids = indices[0].tolist()
        item_scores = scores[0].tolist()
        
        # Filter excluded items
        if exclude_items is not None:
            exclude_set = set(exclude_items)
            filtered_items = []
            filtered_scores = []
            
            for item_id, score in zip(item_ids, item_scores):
                if item_id not in exclude_set:
                    filtered_items.append(item_id)
                    filtered_scores.append(score)
            
            item_ids = filtered_items
            item_scores = filtered_scores
        
        return item_ids, item_scores
    
    def retrieve_for_history(
        self,
        history_items: List[int],
        k: int = 100,
        max_history_length: int = 15,
        exclude_items: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve candidates based on interaction history.
        
        Args:
            history_items: List of item IDs in user's history.
            k: Number of candidates to retrieve.
            max_history_length: Maximum history length to consider.
            exclude_items: Optional list of item IDs to exclude.
            
        Returns:
            Tuple of (item_ids, scores).
        """
        # Truncate history if too long
        if len(history_items) > max_history_length:
            history_items = history_items[-max_history_length:]
        
        # Compute user embedding from history
        user_embedding = self._compute_user_embedding_from_history(history_items)
        
        # Search index
        scores, indices = self.ann_index.search(user_embedding, k=k)
        
        # Extract results
        item_ids = indices[0].tolist()
        item_scores = scores[0].tolist()
        
        # Filter excluded items (including history)
        exclude_set = set(exclude_items) if exclude_items is not None else set()
        exclude_set.update(history_items)
        
        filtered_items = []
        filtered_scores = []
        
        for item_id, score in zip(item_ids, item_scores):
            if item_id not in exclude_set:
                filtered_items.append(item_id)
                filtered_scores.append(score)
        
        return filtered_items, filtered_scores
    
    def retrieve_batch(
        self,
        user_ids: List[int],
        k: int = 100,
        batch_size: int = 256,
    ) -> Dict[int, Tuple[List[int], List[float]]]:
        """
        Retrieve candidates for multiple users in batches.
        
        Args:
            user_ids: List of user IDs.
            k: Number of candidates per user.
            batch_size: Batch size for processing.
            
        Returns:
            Dictionary mapping user IDs to (item_ids, scores) tuples.
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i + batch_size]
            
            # Get user embeddings
            with torch.no_grad():
                user_tensor = torch.tensor(batch_user_ids, device=self.device)
                user_embeddings = self.model.user_tower(user_tensor)
                user_embeddings = user_embeddings.cpu().numpy()
            
            # Search index
            scores, indices = self.ann_index.search(user_embeddings, k=k)
            
            # Store results
            for j, user_id in enumerate(batch_user_ids):
                item_ids = indices[j].tolist()
                item_scores = scores[j].tolist()
                results[user_id] = (item_ids, item_scores)
        
        return results
    
    def retrieve_from_sequences(
        self,
        sequences_df: pd.DataFrame,
        k: int = 100,
        max_history_length: int = 15,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Retrieve candidates from sequence data.
        
        Args:
            sequences_df: DataFrame with user sequences.
            k: Number of candidates per sequence.
            max_history_length: Maximum history length.
            batch_size: Batch size for processing.
            
        Returns:
            DataFrame with candidate lists.
        """
        logger.info(f"Retrieving candidates for {len(sequences_df)} sequences")
        
        candidates = []
        
        # Process in batches
        for i in range(0, len(sequences_df), batch_size):
            batch_df = sequences_df.iloc[i:i + batch_size]
            batch_candidates = self._process_sequence_batch(
                batch_df, k, max_history_length
            )
            candidates.extend(batch_candidates)
        
        # Create result DataFrame
        result_df = pd.DataFrame(candidates)
        logger.info(f"Generated {len(result_df)} candidate lists")
        
        return result_df
    
    def _process_sequence_batch(
        self,
        batch_df: pd.DataFrame,
        k: int,
        max_history_length: int,
    ) -> List[Dict]:
        """Process a batch of sequences to generate candidates."""
        batch_results = []
        
        # Parse histories and compute user embeddings
        histories = []
        for _, row in batch_df.iterrows():
            history = parse_history_string(str(row.get('history_idx', '')))
            if len(history) > max_history_length:
                history = history[-max_history_length:]
            histories.append(history)
        
        # Compute user embeddings from histories
        user_embeddings = self._compute_user_embeddings_from_histories(histories)
        
        # Search index
        scores, indices = self.ann_index.search(user_embeddings, k=k)
        
        # Process results
        for i, (_, row) in enumerate(batch_df.iterrows()):
            history = histories[i]
            item_ids = indices[i].tolist()
            item_scores = scores[i].tolist()
            
            # Ensure positive item is included
            positive_item = int(row['pos_item_idx'])
            if positive_item not in item_ids:
                # Replace last item with positive item
                item_ids[-1] = positive_item
                item_scores[-1] = 1.0  # Give it a high score
            
            # Create candidate string
            candidates_str = ' '.join(map(str, item_ids))
            
            batch_results.append({
                'history_idx': str(row.get('history_idx', '')),
                'pos_item_idx': positive_item,
                'cands': candidates_str,
                'ts': str(row.get('ts', '')),
            })
        
        return batch_results
    
    def _compute_user_embedding_from_history(
        self,
        history_items: List[int],
    ) -> np.ndarray:
        """Compute user embedding from history items."""
        if not history_items:
            # Return zero embedding for empty history
            return np.zeros((1, self.ann_index.dimension), dtype=np.float32)
        
        # Get item embeddings
        history_embeddings = self.item_embeddings[history_items]
        
        # Average and normalize
        user_embedding = np.mean(history_embeddings, axis=0, keepdims=True)
        user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
        
        return user_embedding.astype(np.float32)
    
    def _compute_user_embeddings_from_histories(
        self,
        histories: List[List[int]],
    ) -> np.ndarray:
        """Compute user embeddings from multiple histories."""
        batch_size = len(histories)
        user_embeddings = np.zeros(
            (batch_size, self.ann_index.dimension), dtype=np.float32
        )
        
        for i, history in enumerate(histories):
            if history:
                # Get item embeddings for this history
                history_embeddings = self.item_embeddings[history]
                
                # Average and normalize
                user_embedding = np.mean(history_embeddings, axis=0)
                user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
                
                user_embeddings[i] = user_embedding
        
        return user_embeddings


class HybridRetriever:
    """
    Hybrid retrieval system combining multiple retrieval methods.
    
    Combines Two-Tower retrieval with other methods like popularity,
    collaborative filtering, or content-based filtering.
    """
    
    def __init__(
        self,
        retrievers: Dict[str, any],
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            retrievers: Dictionary of retriever name to retriever instance.
            weights: Optional weights for combining retrievers.
        """
        self.retrievers = retrievers
        self.weights = weights or {name: 1.0 for name in retrievers.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: w / total_weight for name, w in self.weights.items()}
    
    def retrieve(
        self,
        user_id: Optional[int] = None,
        history_items: Optional[List[int]] = None,
        k: int = 100,
        **kwargs
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve candidates using hybrid approach.
        
        Args:
            user_id: Optional user ID.
            history_items: Optional history items.
            k: Number of candidates to retrieve.
            **kwargs: Additional arguments for retrievers.
            
        Returns:
            Tuple of (item_ids, scores).
        """
        # Collect candidates from all retrievers
        all_candidates = {}  # item_id -> weighted_score
        
        for name, retriever in self.retrievers.items():
            weight = self.weights[name]
            
            try:
                # Get candidates from this retriever
                if hasattr(retriever, 'retrieve_for_user') and user_id is not None:
                    item_ids, scores = retriever.retrieve_for_user(user_id, k=k, **kwargs)
                elif hasattr(retriever, 'retrieve_for_history') and history_items is not None:
                    item_ids, scores = retriever.retrieve_for_history(history_items, k=k, **kwargs)
                else:
                    continue
                
                # Add weighted scores
                for item_id, score in zip(item_ids, scores):
                    if item_id in all_candidates:
                        all_candidates[item_id] += weight * score
                    else:
                        all_candidates[item_id] = weight * score
            
            except Exception as e:
                logger.warning(f"Retriever {name} failed: {e}")
                continue
        
        # Sort by combined score and return top-k
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        item_ids = [item_id for item_id, _ in sorted_candidates]
        scores = [score for _, score in sorted_candidates]
        
        return item_ids, scores


def load_retriever(
    model_path: Union[str, Path],
    index_path: Union[str, Path],
    embeddings_path: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
) -> TwoTowerRetriever:
    """
    Load a trained Two-Tower retriever from disk.
    
    Args:
        model_path: Path to saved model state dict.
        index_path: Path to saved ANN index.
        embeddings_path: Path to item embeddings.
        device: Device for model inference.
        
    Returns:
        Loaded TwoTowerRetriever instance.
    """
    # Load model (this would need model architecture info)
    # For now, assume model is loaded elsewhere
    model = None  # This needs to be implemented based on saved model format
    
    # Load index
    ann_index = ANNIndex.load(index_path)
    
    # Load item embeddings
    item_embeddings = read_numpy(embeddings_path)
    
    # Create retriever
    retriever = TwoTowerRetriever(
        model=model,
        ann_index=ann_index,
        item_embeddings=item_embeddings,
        device=device,
    )
    
    return retriever


def evaluate_retrieval(
    retriever: TwoTowerRetriever,
    test_sequences: pd.DataFrame,
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.
    
    Args:
        retriever: Retriever to evaluate.
        test_sequences: Test sequences DataFrame.
        k_values: List of K values to evaluate.
        
    Returns:
        Dictionary with retrieval metrics.
    """
    logger.info(f"Evaluating retrieval on {len(test_sequences)} sequences")
    
    metrics = {}
    
    for k in k_values:
        hits = 0
        total = 0
        
        for _, row in test_sequences.iterrows():
            history = parse_history_string(str(row.get('history_idx', '')))
            positive_item = int(row['pos_item_idx'])
            
            # Retrieve candidates
            candidates, _ = retriever.retrieve_for_history(
                history, k=k, exclude_items=history
            )
            
            # Check if positive item is in candidates
            if positive_item in candidates:
                hits += 1
            
            total += 1
        
        # Compute recall@k
        recall = hits / max(total, 1)
        metrics[f'recall@{k}'] = recall
        
        logger.info(f"Recall@{k}: {recall:.4f}")
    
    return metrics

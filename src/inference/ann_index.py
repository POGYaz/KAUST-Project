"""
Approximate Nearest Neighbor (ANN) index implementation for efficient retrieval.

This module provides FAISS-based indexing for fast similarity search
over large collections of item embeddings, supporting both exact
and approximate search methods.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.io import read_numpy, write_numpy
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Try to import FAISS, fall back to exact search if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to exact search")


class ANNIndex:
    """
    Approximate Nearest Neighbor index for efficient similarity search.
    
    Provides a unified interface for both FAISS-based approximate search
    and exact search using numpy operations.
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = 'flat',
        metric: str = 'cosine',
        n_lists: Optional[int] = None,
        n_probe: Optional[int] = None,
    ):
        """
        Initialize the ANN index.
        
        Args:
            dimension: Embedding dimension.
            index_type: Type of index ('flat', 'ivf', 'hnsw').
            metric: Distance metric ('cosine', 'l2').
            n_lists: Number of clusters for IVF index.
            n_probe: Number of clusters to search in IVF index.
        """
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.n_lists = n_lists
        self.n_probe = n_probe
        
        # Initialize index
        self.index = None
        self.embeddings = None
        self.is_trained = False
        
        if FAISS_AVAILABLE:
            self._create_faiss_index()
        else:
            logger.info("Using exact search (FAISS not available)")
    
    def _create_faiss_index(self) -> None:
        """Create FAISS index based on configuration."""
        if not FAISS_AVAILABLE:
            return
        
        # Choose metric
        if self.metric == 'cosine':
            # For cosine similarity, we'll normalize embeddings and use L2
            metric_type = faiss.METRIC_L2
        elif self.metric == 'l2':
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create index based on type
        if self.index_type == 'flat':
            # Exact search index
            self.index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == 'ivf':
            # IVF (Inverted File) index for approximate search
            n_lists = self.n_lists or min(4096, int(np.sqrt(100000)))  # Heuristic
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_lists, metric_type)
            
            # Set search parameters
            if self.n_probe is not None:
                self.index.nprobe = self.n_probe
        
        elif self.index_type == 'hnsw':
            # HNSW (Hierarchical Navigable Small World) index
            m = 16  # Number of connections for each node
            self.index = faiss.IndexHNSWFlat(self.dimension, m, metric_type)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created FAISS {self.index_type} index with dimension {self.dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embedding matrix of shape (n_items, dimension).
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Store embeddings for exact search fallback
        self.embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            embeddings = self._normalize_embeddings(embeddings)
        
        if FAISS_AVAILABLE and self.index is not None:
            # Train index if necessary
            if not self.is_trained and hasattr(self.index, 'train'):
                logger.info("Training FAISS index...")
                self.index.train(embeddings)
                self.is_trained = True
            
            # Add embeddings to index
            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
        
        else:
            logger.info(f"Stored {len(embeddings)} embeddings for exact search")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings of shape (n_queries, dimension).
            k: Number of nearest neighbors to return.
            
        Returns:
            Tuple of (distances, indices) arrays.
        """
        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            query_embeddings = self._normalize_embeddings(query_embeddings)
        
        if FAISS_AVAILABLE and self.index is not None and self.embeddings is not None:
            # Use FAISS search
            distances, indices = self.index.search(query_embeddings, k)
            
            # Convert L2 distances to cosine similarities if needed
            if self.metric == 'cosine':
                # For normalized vectors: cosine_sim = 1 - l2_dist^2 / 2
                distances = 1.0 - distances / 2.0
            
            return distances, indices
        
        else:
            # Use exact search
            return self._exact_search(query_embeddings, k)
    
    def _exact_search(
        self,
        query_embeddings: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform exact search using numpy operations."""
        if self.embeddings is None:
            raise ValueError("No embeddings added to index")
        
        n_queries = query_embeddings.shape[0]
        n_items = self.embeddings.shape[0]
        k = min(k, n_items)
        
        # Compute similarities
        if self.metric == 'cosine':
            # Cosine similarity
            similarities = query_embeddings @ self.embeddings.T
        elif self.metric == 'l2':
            # Negative L2 distance (for top-k selection)
            similarities = -np.linalg.norm(
                query_embeddings[:, None, :] - self.embeddings[None, :, :],
                axis=2
            )
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Get top-k indices and scores
        top_k_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        
        # Sort within top-k
        indices = np.zeros((n_queries, k), dtype=np.int64)
        distances = np.zeros((n_queries, k), dtype=np.float32)
        
        for i in range(n_queries):
            top_k_for_query = top_k_indices[i]
            scores_for_query = similarities[i, top_k_for_query]
            
            # Sort in descending order of similarity
            sorted_idx = np.argsort(scores_for_query)[::-1]
            indices[i] = top_k_for_query[sorted_idx]
            distances[i] = scores_for_query[sorted_idx]
        
        return distances, indices
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save the index to disk.
        
        Args:
            save_path: Path to save the index.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'n_lists': self.n_lists,
            'n_probe': self.n_probe,
            'is_trained': self.is_trained,
        }
        
        config_path = save_path.with_suffix('.config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        # Save embeddings
        if self.embeddings is not None:
            embeddings_path = save_path.with_suffix('.embeddings.npy')
            write_numpy(self.embeddings, embeddings_path)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            index_path = save_path.with_suffix('.faiss')
            faiss.write_index(self.index, str(index_path))
        
        logger.info(f"Index saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'ANNIndex':
        """
        Load an index from disk.
        
        Args:
            load_path: Path to load the index from.
            
        Returns:
            Loaded ANNIndex instance.
        """
        load_path = Path(load_path)
        
        # Load configuration
        config_path = load_path.with_suffix('.config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Create index instance
        index = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            metric=config['metric'],
            n_lists=config['n_lists'],
            n_probe=config['n_probe'],
        )
        
        index.is_trained = config['is_trained']
        
        # Load embeddings
        embeddings_path = load_path.with_suffix('.embeddings.npy')
        if embeddings_path.exists():
            index.embeddings = read_numpy(embeddings_path)
        
        # Load FAISS index
        if FAISS_AVAILABLE:
            index_path = load_path.with_suffix('.faiss')
            if index_path.exists():
                index.index = faiss.read_index(str(index_path))
        
        logger.info(f"Index loaded from {load_path}")
        return index
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics.
        """
        stats = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.is_trained,
        }
        
        if self.embeddings is not None:
            stats['n_embeddings'] = len(self.embeddings)
        
        if FAISS_AVAILABLE and self.index is not None:
            stats['faiss_available'] = True
            stats['faiss_ntotal'] = self.index.ntotal
        else:
            stats['faiss_available'] = False
        
        return stats


def build_index_from_embeddings(
    embeddings: np.ndarray,
    index_type: str = 'flat',
    metric: str = 'cosine',
    **kwargs
) -> ANNIndex:
    """
    Build an ANN index from embeddings.
    
    Args:
        embeddings: Embedding matrix of shape (n_items, dimension).
        index_type: Type of index to build.
        metric: Distance metric to use.
        **kwargs: Additional arguments for index configuration.
        
    Returns:
        Built and populated ANNIndex.
    """
    dimension = embeddings.shape[1]
    
    # Create index
    index = ANNIndex(
        dimension=dimension,
        index_type=index_type,
        metric=metric,
        **kwargs
    )
    
    # Add embeddings
    index.add_embeddings(embeddings)
    
    return index


def search_index(
    index: ANNIndex,
    query_embeddings: np.ndarray,
    k: int = 10,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search index with batching for large query sets.
    
    Args:
        index: ANN index to search.
        query_embeddings: Query embeddings.
        k: Number of neighbors to return.
        batch_size: Batch size for processing queries.
        
    Returns:
        Tuple of (distances, indices) for all queries.
    """
    n_queries = query_embeddings.shape[0]
    
    if n_queries <= batch_size:
        # Process all queries at once
        return index.search(query_embeddings, k)
    
    # Process in batches
    all_distances = []
    all_indices = []
    
    for i in range(0, n_queries, batch_size):
        batch_queries = query_embeddings[i:i + batch_size]
        batch_distances, batch_indices = index.search(batch_queries, k)
        
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)
    
    # Concatenate results
    distances = np.concatenate(all_distances, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    return distances, indices

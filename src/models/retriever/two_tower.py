"""
Two-Tower neural retrieval model implementation.

This module implements the Two-Tower architecture for neural information
retrieval, with separate user and item towers that produce embeddings
for efficient similarity computation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logging import get_logger

logger = get_logger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and dropout.
    
    Implements a residual connection with two linear layers,
    layer normalization, and ReLU activation.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize the residual block.
        
        Args:
            d_model: Model dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, d_model).
            
        Returns:
            Output tensor of shape (batch_size, d_model).
        """
        # First transformation
        h = self.dropout(F.relu(self.ln1(self.fc1(x))))
        
        # Second transformation
        h = self.dropout(F.relu(self.ln2(self.fc2(h))))
        
        # Residual connection
        return F.relu(x + h)


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for neural information retrieval.
    
    Implements separate user and item towers with residual blocks
    and layer normalization. The towers produce normalized embeddings
    suitable for cosine similarity computation.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        d_model: int = 256,
        dropout: float = 0.2,
        n_blocks: int = 2,
        embedding_init_std: float = 0.1,
    ):
        """
        Initialize the Two-Tower model.
        
        Args:
            n_users: Number of users in the system.
            n_items: Number of items in the system.
            d_model: Embedding dimension.
            dropout: Dropout probability.
            n_blocks: Number of residual blocks per tower.
            embedding_init_std: Standard deviation for embedding initialization.
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.d_model = d_model
        
        # User tower
        self.user_embedding = nn.Embedding(n_users, d_model)
        self.user_blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.user_norm = nn.LayerNorm(d_model)
        
        # Item tower
        self.item_embedding = nn.Embedding(n_items, d_model)
        self.item_blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.item_norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings
        self._init_embeddings(embedding_init_std)
    
    def _init_embeddings(self, std: float) -> None:
        """Initialize embedding layers with normal distribution."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=std)
    
    def user_tower(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the user tower.
        
        Args:
            user_ids: User ID tensor of shape (batch_size,).
            
        Returns:
            Normalized user embeddings of shape (batch_size, d_model).
        """
        # Get user embeddings
        x = self.user_embedding(user_ids)
        
        # Apply residual blocks
        for block in self.user_blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.user_norm(x)
        
        # L2 normalization for cosine similarity
        return F.normalize(x, p=2, dim=1)
    
    def item_tower(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the item tower.
        
        Args:
            item_ids: Item ID tensor of shape (batch_size,).
            
        Returns:
            Normalized item embeddings of shape (batch_size, d_model).
        """
        # Get item embeddings
        x = self.item_embedding(item_ids)
        
        # Apply residual blocks
        for block in self.item_blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.item_norm(x)
        
        # L2 normalization for cosine similarity
        return F.normalize(x, p=2, dim=1)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for training with positive and negative items.
        
        Args:
            user_ids: User ID tensor of shape (batch_size,).
            positive_item_ids: Positive item ID tensor of shape (batch_size,).
            negative_item_ids: Optional negative item ID tensor of shape (batch_size, n_negatives).
            
        Returns:
            Tuple of (positive_scores, negative_scores) if negatives provided,
            otherwise just positive_scores.
        """
        # Get user embeddings
        user_embeddings = self.user_tower(user_ids)
        
        # Get positive item embeddings
        positive_embeddings = self.item_tower(positive_item_ids)
        
        # Compute positive scores (cosine similarity)
        positive_scores = (user_embeddings * positive_embeddings).sum(dim=1)
        
        if negative_item_ids is not None:
            # Get negative item embeddings
            batch_size, n_negatives = negative_item_ids.shape
            negative_embeddings = self.item_tower(negative_item_ids.view(-1))
            negative_embeddings = negative_embeddings.view(batch_size, n_negatives, self.d_model)
            
            # Compute negative scores
            negative_scores = torch.sum(
                user_embeddings.unsqueeze(1) * negative_embeddings, dim=2
            )
            
            return positive_scores, negative_scores
        
        return positive_scores
    
    def get_all_user_embeddings(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get embeddings for all users.
        
        Args:
            device: Device to place embeddings on.
            
        Returns:
            User embeddings tensor of shape (n_users, d_model).
        """
        if device is None:
            device = next(self.parameters()).device
        
        user_ids = torch.arange(self.n_users, device=device)
        return self.user_tower(user_ids)
    
    def get_all_item_embeddings(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get embeddings for all items.
        
        Args:
            device: Device to place embeddings on.
            
        Returns:
            Item embeddings tensor of shape (n_items, d_model).
        """
        if device is None:
            device = next(self.parameters()).device
        
        item_ids = torch.arange(self.n_items, device=device)
        return self.item_tower(item_ids)
    
    def compute_similarity_matrix(
        self,
        user_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the full user-item similarity matrix.
        
        Args:
            user_embeddings: Optional precomputed user embeddings.
            item_embeddings: Optional precomputed item embeddings.
            
        Returns:
            Similarity matrix of shape (n_users, n_items).
        """
        if user_embeddings is None:
            user_embeddings = self.get_all_user_embeddings()
        
        if item_embeddings is None:
            item_embeddings = self.get_all_item_embeddings()
        
        return user_embeddings @ item_embeddings.T
    
    def recommend_for_user(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate top-k recommendations for a single user.
        
        Args:
            user_id: User ID to generate recommendations for.
            k: Number of recommendations to generate.
            exclude_items: Optional tensor of item IDs to exclude.
            item_embeddings: Optional precomputed item embeddings.
            
        Returns:
            Tensor of recommended item IDs of shape (k,).
        """
        device = next(self.parameters()).device
        
        # Get user embedding
        user_tensor = torch.tensor([user_id], device=device)
        user_embedding = self.user_tower(user_tensor)
        
        # Get item embeddings
        if item_embeddings is None:
            item_embeddings = self.get_all_item_embeddings()
        
        # Compute similarities
        similarities = user_embedding @ item_embeddings.T
        similarities = similarities.squeeze(0)
        
        # Exclude specified items
        if exclude_items is not None:
            similarities[exclude_items] = float('-inf')
        
        # Get top-k recommendations
        _, top_k_indices = torch.topk(similarities, k=k)
        
        return top_k_indices


class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for Two-Tower model training with sequence data.
    
    Handles loading of user sequences with history and positive items,
    and generates negative samples on-the-fly.
    """
    
    def __init__(
        self,
        sequences_df,
        n_items: int,
        n_negatives: int = 50,
        random_seed: int = 42,
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            sequences_df: DataFrame with user sequences.
            n_items: Total number of items for negative sampling.
            n_negatives: Number of negative samples per positive.
            random_seed: Random seed for reproducibility.
        """
        self.sequences_df = sequences_df
        self.n_items = n_items
        self.n_negatives = n_negatives
        
        # Initialize random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(random_seed)
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences_df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Dictionary with user, positive item, negative items, and history.
        """
        row = self.sequences_df.iloc[idx]
        
        user_id = torch.tensor(int(row['user_idx']), dtype=torch.long)
        positive_item_id = torch.tensor(int(row['pos_item_idx']), dtype=torch.long)
        
        # Parse history
        history = []
        if hasattr(row, 'history_idx') and row['history_idx']:
            try:
                history = [int(x) for x in str(row['history_idx']).split()]
            except (ValueError, AttributeError):
                history = []
        
        history_tensor = torch.tensor(history, dtype=torch.long)
        
        # Generate negative samples
        # Exclude positive item and history items from negative sampling
        forbidden_items = set(history + [int(positive_item_id)])
        available_items = list(set(range(self.n_items)) - forbidden_items)
        
        if len(available_items) >= self.n_negatives:
            # Sample without replacement
            negative_indices = torch.randperm(
                len(available_items), generator=self.rng
            )[:self.n_negatives]
            negative_items = torch.tensor(
                [available_items[i] for i in negative_indices], dtype=torch.long
            )
        else:
            # Sample with replacement if not enough items
            negative_items = torch.tensor(
                available_items * ((self.n_negatives // len(available_items)) + 1),
                dtype=torch.long
            )[:self.n_negatives]
        
        return {
            'user': user_id,
            'positive': positive_item_id,
            'negatives': negative_items,
            'history': history_tensor,
        }


def collate_sequences(batch: list) -> dict:
    """
    Collate function for sequence dataset batches.
    
    Args:
        batch: List of dictionaries from SequenceDataset.__getitem__.
        
    Returns:
        Batched dictionary with padded sequences.
    """
    users = torch.stack([item['user'] for item in batch])
    positives = torch.stack([item['positive'] for item in batch])
    negatives = torch.stack([item['negatives'] for item in batch])
    
    # Pad history sequences
    histories = [item['history'] for item in batch]
    max_history_length = max(len(hist) for hist in histories) if histories else 0
    
    if max_history_length > 0:
        padded_histories = torch.full(
            (len(batch), max_history_length), -1, dtype=torch.long
        )
        
        for i, hist in enumerate(histories):
            if len(hist) > 0:
                padded_histories[i, :len(hist)] = hist
        
        # Create mask for valid history positions
        history_mask = (padded_histories >= 0)
    else:
        padded_histories = torch.empty((len(batch), 0), dtype=torch.long)
        history_mask = torch.empty((len(batch), 0), dtype=torch.bool)
    
    return {
        'users': users,
        'positives': positives,
        'negatives': negatives,
        'histories': padded_histories,
        'history_mask': history_mask,
    }

"""
Loss functions for neural retrieval models.

This module implements various loss functions commonly used in
neural information retrieval, including InfoNCE, BPR, and
contrastive losses with temperature scaling and regularization.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logging import get_logger

logger = get_logger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss.
    
    Implements the InfoNCE loss commonly used in contrastive learning
    and neural retrieval. The loss maximizes agreement between positive
    pairs while minimizing agreement with negative samples.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_q_correction: bool = False,
        reduction: str = 'mean',
    ):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities.
            use_q_correction: Whether to apply q-correction for negative sampling bias.
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        
        self.temperature = temperature
        self.use_q_correction = use_q_correction
        self.reduction = reduction
        
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            user_embeddings: User embeddings of shape (batch_size, d_model).
            positive_embeddings: Positive item embeddings of shape (batch_size, d_model).
            negative_embeddings: Negative item embeddings of shape (batch_size, n_negatives, d_model).
            
        Returns:
            InfoNCE loss tensor.
        """
        batch_size, d_model = user_embeddings.shape
        n_negatives = negative_embeddings.size(1)
        
        # Scale by temperature
        scale = 1.0 / max(self.temperature, 1e-6)
        
        # Compute positive similarities
        positive_similarities = (user_embeddings * positive_embeddings).sum(dim=1) * scale
        
        # Compute negative similarities
        negative_similarities = torch.sum(
            user_embeddings.unsqueeze(1) * negative_embeddings, dim=2
        ) * scale
        
        # Concatenate positive and negative similarities
        # Shape: (batch_size, 1 + n_negatives)
        all_similarities = torch.cat([
            positive_similarities.unsqueeze(1),
            negative_similarities
        ], dim=1)
        
        # Apply q-correction if enabled
        if self.use_q_correction:
            q_correction = torch.zeros_like(all_similarities)
            # Only correct negatives (leave positives unchanged)
            q_correction[:, 1:] = -math.log(n_negatives)
            all_similarities = all_similarities + q_correction
        
        # Targets are always the first position (positive)
        targets = torch.zeros(batch_size, dtype=torch.long, device=user_embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(all_similarities, targets, reduction=self.reduction)
        
        return loss


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss.
    
    Implements the BPR loss which assumes that users prefer observed
    items over unobserved items. The loss maximizes the difference
    between positive and negative item scores.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize BPR loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            positive_scores: Scores for positive items of shape (batch_size,).
            negative_scores: Scores for negative items of shape (batch_size, n_negatives).
            
        Returns:
            BPR loss tensor.
        """
        # Expand positive scores to match negative scores shape
        positive_scores_expanded = positive_scores.unsqueeze(1)  # (batch_size, 1)
        
        # Compute pairwise differences
        score_differences = positive_scores_expanded - negative_scores  # (batch_size, n_negatives)
        
        # Apply sigmoid and take negative log
        bpr_loss = -F.logsigmoid(score_differences)
        
        # Apply reduction
        if self.reduction == 'mean':
            return bpr_loss.mean()
        elif self.reduction == 'sum':
            return bpr_loss.sum()
        else:
            return bpr_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for similarity learning.
    
    Implements contrastive loss that pulls positive pairs together
    and pushes negative pairs apart with a margin.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs.
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            user_embeddings: User embeddings of shape (batch_size, d_model).
            positive_embeddings: Positive item embeddings of shape (batch_size, d_model).
            negative_embeddings: Negative item embeddings of shape (batch_size, n_negatives, d_model).
            
        Returns:
            Contrastive loss tensor.
        """
        # Compute positive distances (should be small)
        positive_distances = F.pairwise_distance(user_embeddings, positive_embeddings)
        positive_loss = positive_distances.pow(2)
        
        # Compute negative distances (should be large)
        batch_size, n_negatives, d_model = negative_embeddings.shape
        user_expanded = user_embeddings.unsqueeze(1).expand(-1, n_negatives, -1)
        
        negative_distances = F.pairwise_distance(
            user_expanded.reshape(-1, d_model),
            negative_embeddings.reshape(-1, d_model)
        ).reshape(batch_size, n_negatives)
        
        negative_loss = F.relu(self.margin - negative_distances).pow(2)
        
        # Combine losses
        total_loss = positive_loss + negative_loss.sum(dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for embedding learning.
    
    Implements triplet loss that ensures the distance between anchor
    and positive is smaller than the distance between anchor and
    negative by at least a margin.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet constraint.
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor_embeddings: Anchor (user) embeddings of shape (batch_size, d_model).
            positive_embeddings: Positive item embeddings of shape (batch_size, d_model).
            negative_embeddings: Negative item embeddings of shape (batch_size, n_negatives, d_model).
            
        Returns:
            Triplet loss tensor.
        """
        # Compute positive distances
        positive_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        
        # Compute negative distances
        batch_size, n_negatives, d_model = negative_embeddings.shape
        anchor_expanded = anchor_embeddings.unsqueeze(1).expand(-1, n_negatives, -1)
        
        negative_distances = F.pairwise_distance(
            anchor_expanded.reshape(-1, d_model),
            negative_embeddings.reshape(-1, d_model)
        ).reshape(batch_size, n_negatives)
        
        # Compute triplet loss for each negative
        positive_distances_expanded = positive_distances.unsqueeze(1)
        triplet_losses = F.relu(
            positive_distances_expanded - negative_distances + self.margin
        )
        
        # Take the mean over negatives, then apply reduction
        triplet_loss = triplet_losses.mean(dim=1)
        
        if self.reduction == 'mean':
            return triplet_loss.mean()
        elif self.reduction == 'sum':
            return triplet_loss.sum()
        else:
            return triplet_loss


class SampledSoftmaxLoss(nn.Module):
    """
    Sampled softmax loss for large vocabulary problems.
    
    Implements sampled softmax loss which approximates the full softmax
    by only computing scores for a subset of negative samples, making
    it efficient for large item catalogs.
    """
    
    def __init__(
        self,
        num_classes: int,
        num_sampled: int,
        temperature: float = 1.0,
    ):
        """
        Initialize sampled softmax loss.
        
        Args:
            num_classes: Total number of classes (items).
            num_sampled: Number of negative samples.
            temperature: Temperature for scaling.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.temperature = temperature
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sampled softmax loss.
        
        Args:
            user_embeddings: User embeddings of shape (batch_size, d_model).
            item_embeddings: Full item embedding matrix of shape (num_items, d_model).
            positive_item_ids: Positive item IDs of shape (batch_size,).
            negative_item_ids: Negative item IDs of shape (batch_size, num_sampled).
            
        Returns:
            Sampled softmax loss tensor.
        """
        batch_size = user_embeddings.size(0)
        
        # Get positive item embeddings
        positive_embeddings = item_embeddings[positive_item_ids]
        
        # Get negative item embeddings
        negative_embeddings = item_embeddings[negative_item_ids.view(-1)].view(
            batch_size, self.num_sampled, -1
        )
        
        # Compute similarities
        positive_similarities = (user_embeddings * positive_embeddings).sum(dim=1) / self.temperature
        negative_similarities = torch.sum(
            user_embeddings.unsqueeze(1) * negative_embeddings, dim=2
        ) / self.temperature
        
        # Concatenate similarities
        all_similarities = torch.cat([
            positive_similarities.unsqueeze(1),
            negative_similarities
        ], dim=1)
        
        # Targets are always the first position
        targets = torch.zeros(batch_size, dtype=torch.long, device=user_embeddings.device)
        
        # Compute cross-entropy loss
        return F.cross_entropy(all_similarities, targets)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining multiple objectives.
    
    Combines different loss functions with learnable or fixed weights
    for multi-task learning scenarios.
    """
    
    def __init__(
        self,
        loss_functions: dict,
        loss_weights: Optional[dict] = None,
        learnable_weights: bool = False,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            loss_functions: Dictionary mapping loss names to loss functions.
            loss_weights: Optional dictionary of loss weights.
            learnable_weights: Whether to make loss weights learnable parameters.
        """
        super().__init__()
        
        self.loss_functions = nn.ModuleDict(loss_functions)
        
        if learnable_weights:
            # Initialize learnable weights
            self.loss_weights = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(1.0))
                for name in loss_functions.keys()
            })
        else:
            # Use fixed weights
            if loss_weights is None:
                loss_weights = {name: 1.0 for name in loss_functions.keys()}
            
            self.register_buffer('_loss_weights', torch.tensor(list(loss_weights.values())))
            self.loss_names = list(loss_weights.keys())
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss.
        
        Args:
            *args: Arguments to pass to loss functions.
            **kwargs: Keyword arguments to pass to loss functions.
            
        Returns:
            Tuple of (total_loss, individual_losses).
        """
        individual_losses = {}
        total_loss = 0.0
        
        for i, (name, loss_fn) in enumerate(self.loss_functions.items()):
            # Compute individual loss
            loss_value = loss_fn(*args, **kwargs)
            individual_losses[name] = loss_value
            
            # Get weight
            if hasattr(self, 'loss_weights'):
                weight = self.loss_weights[name]
            else:
                weight = self._loss_weights[i]
            
            # Add to total loss
            total_loss = total_loss + weight * loss_value
        
        return total_loss, individual_losses


def create_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type: Type of loss function to create.
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        Initialized loss function.
        
    Raises:
        ValueError: If loss_type is not recognized.
    """
    loss_registry = {
        'infonce': InfoNCELoss,
        'bpr': BPRLoss,
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'sampled_softmax': SampledSoftmaxLoss,
    }
    
    if loss_type.lower() not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available losses: {available_losses}"
        )
    
    loss_class = loss_registry[loss_type.lower()]
    return loss_class(**kwargs)

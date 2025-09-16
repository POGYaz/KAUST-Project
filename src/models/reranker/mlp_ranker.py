"""
MLP-based reranking model implementation.

This module implements neural reranking models using multi-layer
perceptrons (MLPs) with various regularization techniques and
feature engineering capabilities.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logging import get_logger

logger = get_logger(__name__)


class FeatureDropout(nn.Module):
    """
    Feature-specific dropout layer.
    
    Applies dropout to specific features during training to reduce
    overfitting and improve generalization. Useful for regularizing
    strong features like dot products.
    """
    
    def __init__(self, feature_indices: List[int], dropout_prob: float = 0.3):
        """
        Initialize feature dropout.
        
        Args:
            feature_indices: List of feature indices to apply dropout to.
            dropout_prob: Probability of dropping features.
        """
        super().__init__()
        self.feature_indices = feature_indices
        self.dropout_prob = dropout_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature dropout.
        
        Args:
            x: Input features of shape (batch_size, n_features).
            
        Returns:
            Features with dropout applied.
        """
        if not self.training or self.dropout_prob <= 0:
            return x
        
        # Create dropout mask for specified features
        mask = torch.ones_like(x)
        
        for feature_idx in self.feature_indices:
            if feature_idx < x.size(1):
                # Apply dropout to this feature
                feature_mask = (torch.rand(x.size(0), device=x.device) > self.dropout_prob).float()
                mask[:, feature_idx] = feature_mask
        
        return x * mask


class MLPRanker(nn.Module):
    """
    Multi-layer perceptron for reranking candidates.
    
    Implements a deep MLP with residual connections, layer normalization,
    and various regularization techniques for ranking candidate items.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [384, 384, 192],
        dropout: float = 0.3,
        feature_dropout_indices: Optional[List[int]] = None,
        feature_dropout_prob: float = 0.3,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        activation: str = 'relu',
    ):
        """
        Initialize the MLP ranker.
        
        Args:
            input_dim: Dimension of input features.
            hidden_dims: List of hidden layer dimensions.
            dropout: Standard dropout probability.
            feature_dropout_indices: Indices of features to apply feature dropout.
            feature_dropout_prob: Probability for feature dropout.
            use_residual: Whether to use residual connections.
            use_layer_norm: Whether to use layer normalization.
            activation: Activation function name ('relu', 'gelu', 'swish').
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Feature dropout
        if feature_dropout_indices is not None:
            self.feature_dropout = FeatureDropout(
                feature_dropout_indices, feature_dropout_prob
            )
        else:
            self.feature_dropout = nn.Identity()
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer normalization
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.layer_norms.append(nn.Identity())
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation.lower()]
    
    def _init_weights(self):
        """Initialize model weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP ranker.
        
        Args:
            x: Input features of shape (batch_size, input_dim).
            
        Returns:
            Ranking scores of shape (batch_size,).
        """
        # Apply feature dropout
        x = self.feature_dropout(x)
        
        # Forward through hidden layers
        for i, (layer, layer_norm, dropout) in enumerate(
            zip(self.layers, self.layer_norms, self.dropouts)
        ):
            # Store input for potential residual connection
            residual = x if self.use_residual and x.size(-1) == self.hidden_dims[i] else None
            
            # Linear transformation
            x = layer(x)
            
            # Layer normalization
            x = layer_norm(x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = dropout(x)
            
            # Residual connection
            if residual is not None:
                x = x + residual
        
        # Output layer
        output = self.output_layer(x)
        
        return output.squeeze(-1)
    
    def predict_scores(
        self,
        features: torch.Tensor,
        batch_size: int = 1000,
    ) -> torch.Tensor:
        """
        Predict ranking scores for a large number of candidates.
        
        Args:
            features: Feature tensor of shape (n_candidates, input_dim).
            batch_size: Batch size for processing.
            
        Returns:
            Ranking scores of shape (n_candidates,).
        """
        self.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, features.size(0), batch_size):
                batch_features = features[i:i + batch_size]
                batch_scores = self.forward(batch_features)
                scores.append(batch_scores)
        
        return torch.cat(scores, dim=0)
    
    def rank_candidates(
        self,
        features: torch.Tensor,
        k: int = 10,
        return_scores: bool = False,
    ) -> torch.Tensor:
        """
        Rank candidates and return top-k.
        
        Args:
            features: Feature tensor of shape (n_candidates, input_dim).
            k: Number of top candidates to return.
            return_scores: Whether to return scores along with indices.
            
        Returns:
            Top-k candidate indices, optionally with scores.
        """
        scores = self.predict_scores(features)
        
        # Get top-k indices
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
        
        if return_scores:
            return top_k_indices, top_k_scores
        else:
            return top_k_indices


class DeepCrossNetwork(nn.Module):
    """
    Deep & Cross Network (DCN) for feature interactions.
    
    Implements DCN which explicitly models feature interactions
    through cross layers while maintaining deep learning capabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        cross_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize Deep & Cross Network.
        
        Args:
            input_dim: Dimension of input features.
            hidden_dims: Hidden dimensions for deep part.
            cross_layers: Number of cross layers.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.cross_layers = cross_layers
        
        # Cross network
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1))
            for _ in range(cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(cross_layers)
        ])
        
        # Deep network
        self.deep_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final layer
        self.output_layer = nn.Linear(input_dim + prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DCN.
        
        Args:
            x: Input features of shape (batch_size, input_dim).
            
        Returns:
            Output scores of shape (batch_size,).
        """
        # Cross network
        cross_output = x
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            # Cross layer: x_l+1 = x_0 * (W_l * x_l + b_l) + x_l
            xl_w = torch.matmul(cross_output, weight)  # (batch_size, 1)
            cross_output = x * xl_w + bias + cross_output
        
        # Deep network
        deep_output = x
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        
        # Concatenate cross and deep outputs
        combined = torch.cat([cross_output, deep_output], dim=-1)
        
        # Final output
        output = self.output_layer(combined)
        return output.squeeze(-1)


class WideAndDeep(nn.Module):
    """
    Wide & Deep model for recommendation.
    
    Combines wide linear models with deep neural networks
    for both memorization and generalization.
    """
    
    def __init__(
        self,
        wide_dim: int,
        deep_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Initialize Wide & Deep model.
        
        Args:
            wide_dim: Dimension of wide features.
            deep_dim: Dimension of deep features.
            hidden_dims: Hidden dimensions for deep part.
            dropout: Dropout probability.
        """
        super().__init__()
        
        # Wide part (linear)
        self.wide_layer = nn.Linear(wide_dim, 1)
        
        # Deep part
        self.deep_layers = nn.ModuleList()
        prev_dim = deep_dim
        
        for hidden_dim in hidden_dims:
            self.deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.deep_output = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        wide_features: torch.Tensor,
        deep_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through Wide & Deep model.
        
        Args:
            wide_features: Wide features of shape (batch_size, wide_dim).
            deep_features: Deep features of shape (batch_size, deep_dim).
            
        Returns:
            Output scores of shape (batch_size,).
        """
        # Wide part
        wide_output = self.wide_layer(wide_features)
        
        # Deep part
        deep_output = deep_features
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        deep_output = self.deep_output(deep_output)
        
        # Combine wide and deep
        output = wide_output + deep_output
        return output.squeeze(-1)


def create_ranker(
    ranker_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating ranking models.
    
    Args:
        ranker_type: Type of ranker to create.
        input_dim: Input feature dimension.
        **kwargs: Additional arguments for the ranker.
        
    Returns:
        Initialized ranking model.
        
    Raises:
        ValueError: If ranker_type is not recognized.
    """
    ranker_registry = {
        'mlp': MLPRanker,
        'dcn': DeepCrossNetwork,
        'wide_deep': WideAndDeep,
    }
    
    if ranker_type.lower() not in ranker_registry:
        available_rankers = list(ranker_registry.keys())
        raise ValueError(
            f"Unknown ranker type: {ranker_type}. "
            f"Available rankers: {available_rankers}"
        )
    
    ranker_class = ranker_registry[ranker_type.lower()]
    return ranker_class(input_dim=input_dim, **kwargs)

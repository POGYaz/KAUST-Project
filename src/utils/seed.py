"""
Reproducibility utilities for deterministic training and evaluation.

This module provides functions to set random seeds across different libraries
to ensure reproducible results in machine learning experiments.
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results across all libraries.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (if available)
    - Environment variable for Python hash seed
    
    Args:
        seed: Random seed value to use across all libraries.
        
    Note:
        For complete reproducibility, this should be called at the beginning
        of your script before importing any other modules that use randomness.
    """
    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set Python built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Enable deterministic algorithms if supported
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fallback for older PyTorch versions
            pass
            
        # Configure cuDNN for reproducibility
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    except ImportError:
        # PyTorch not available, skip torch-specific seeding
        pass


def get_device() -> str:
    """
    Get the appropriate device for PyTorch computations.
    
    Returns:
        Device string: 'cuda' if CUDA is available, 'cpu' otherwise.
        
    Note:
        Returns 'cpu' if PyTorch is not available.
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def configure_torch_performance() -> None:
    """
    Configure PyTorch for optimal performance.
    
    Sets various PyTorch configurations that can improve training speed
    while maintaining reproducibility when used with set_seed().
    """
    try:
        import torch
        
        # Set float32 matrix multiplication precision for better performance
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
            
        # Enable optimized attention if available (PyTorch 2.0+)
        if hasattr(torch.backends, 'opt_einsum'):
            torch.backends.opt_einsum.enabled = True
            
    except ImportError:
        # PyTorch not available, skip configuration
        pass

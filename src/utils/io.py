"""
Input/Output utilities for the recommendation system.

This module provides standardized functions for reading and writing
various data formats commonly used in the recommendation system,
including Parquet, JSON, and NumPy arrays.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


def ensure_path(path: Union[str, Path]) -> Path:
    """
    Ensure path is a Path object and create parent directories.
    
    Args:
        path: File or directory path.
        
    Returns:
        Path object with parent directories created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_parquet(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    engine: str = "fastparquet",
) -> pd.DataFrame:
    """
    Read Parquet file with error handling.
    
    Args:
        path: Path to Parquet file.
        columns: Optional list of columns to read.
        engine: Parquet engine to use ('fastparquet' or 'pyarrow').
        
    Returns:
        DataFrame loaded from Parquet file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    try:
        return pd.read_parquet(path, columns=columns, engine=engine)
    except Exception as e:
        raise ValueError(f"Error reading Parquet file {path}: {e}")


def write_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    engine: str = "fastparquet",
    index: bool = False,
) -> None:
    """
    Write DataFrame to Parquet file with error handling.
    
    Args:
        df: DataFrame to write.
        path: Output path for Parquet file.
        engine: Parquet engine to use ('fastparquet' or 'pyarrow').
        index: Whether to include the DataFrame index.
        
    Raises:
        ValueError: If the DataFrame cannot be written.
    """
    path = ensure_path(path)
    
    try:
        df.to_parquet(path, engine=engine, index=index)
    except Exception as e:
        raise ValueError(f"Error writing Parquet file {path}: {e}")


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON file with error handling.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        Dictionary loaded from JSON file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file {path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading JSON file {path}: {e}")


def write_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Write dictionary to JSON file with error handling.
    
    Args:
        data: Dictionary to write.
        path: Output path for JSON file.
        indent: Number of spaces for indentation.
        
    Raises:
        ValueError: If the data cannot be serialized to JSON.
    """
    path = ensure_path(path)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Error writing JSON file {path}: {e}")


def read_numpy(path: Union[str, Path]) -> np.ndarray:
    """
    Read NumPy array from file with error handling.
    
    Args:
        path: Path to NumPy file (.npy or .npz).
        
    Returns:
        NumPy array loaded from file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")
    
    try:
        if path.suffix == '.npz':
            # For .npz files, return the first array
            with np.load(path) as data:
                arrays = list(data.values())
                if not arrays:
                    raise ValueError("Empty .npz file")
                return arrays[0]
        else:
            return np.load(path)
    except Exception as e:
        raise ValueError(f"Error reading NumPy file {path}: {e}")


def write_numpy(
    array: np.ndarray,
    path: Union[str, Path],
    compressed: bool = False,
) -> None:
    """
    Write NumPy array to file with error handling.
    
    Args:
        array: NumPy array to write.
        path: Output path for NumPy file.
        compressed: Whether to use compressed format (.npz).
        
    Raises:
        ValueError: If the array cannot be saved.
    """
    path = ensure_path(path)
    
    try:
        if compressed or path.suffix == '.npz':
            np.savez_compressed(path, array)
        else:
            np.save(path, array)
    except Exception as e:
        raise ValueError(f"Error writing NumPy file {path}: {e}")


def read_torch(path: Union[str, Path], map_location: str = "cpu") -> torch.Tensor:
    """
    Read PyTorch tensor from file with error handling.
    
    Args:
        path: Path to PyTorch tensor file.
        map_location: Device to load tensor on.
        
    Returns:
        PyTorch tensor loaded from file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"PyTorch file not found: {path}")
    
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise ValueError(f"Error reading PyTorch file {path}: {e}")


def write_torch(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """
    Write PyTorch tensor to file with error handling.
    
    Args:
        tensor: PyTorch tensor to write.
        path: Output path for tensor file.
        
    Raises:
        ValueError: If the tensor cannot be saved.
    """
    path = ensure_path(path)
    
    try:
        torch.save(tensor, path)
    except Exception as e:
        raise ValueError(f"Error writing PyTorch file {path}: {e}")


def read_pickle(path: Union[str, Path]) -> Any:
    """
    Read pickled object from file with error handling.
    
    Args:
        path: Path to pickle file.
        
    Returns:
        Object loaded from pickle file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error reading pickle file {path}: {e}")


def write_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Write object to pickle file with error handling.
    
    Args:
        obj: Object to pickle.
        path: Output path for pickle file.
        
    Raises:
        ValueError: If the object cannot be pickled.
    """
    path = ensure_path(path)
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise ValueError(f"Error writing pickle file {path}: {e}")


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    List files in directory matching a pattern.
    
    Args:
        directory: Directory to search in.
        pattern: Glob pattern to match files.
        recursive: Whether to search recursively.
        
    Returns:
        List of Path objects matching the pattern.
        
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: Path to file.
        
    Returns:
        File size in bytes.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes.
        
    Returns:
        Formatted file size string (e.g., "1.5 MB").
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

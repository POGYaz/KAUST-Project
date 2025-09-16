"""
Dataset handling and loading utilities for the recommendation system.

This module provides classes and functions for loading, processing, and
managing datasets used in the recommendation system, including the Jarir
retail dataset.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)


class JarirDatasetLoader:
    """
    Loader for the Jarir retail dataset.
    
    Handles loading and basic preprocessing of the raw Jarir Excel dataset,
    including date parsing, column standardization, and basic validation.
    """
    
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """
        Initialize the Jarir dataset loader.
        
        Args:
            raw_data_path: Path to the raw Jarir Excel file.
            output_dir: Directory to save processed data.
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Column mapping from raw Excel to standardized names
        self.column_mapping = {
            'ItemNumber': 'stock_code',
            'ItemDescription': 'description',
            'CustomerId': 'customer_id',
            'Sales Quantity 2024': 'quantity',
            'Sales Amount 2024': 'line_amount',
            'Date': 'invoice_date',
            'Showroom': 'country',
            'Brand': 'brand',
            'GL Class': 'category',
            'Vendor Prefix': 'vendor',
            'Model': 'model',
            'ShortItemNo': 'short_item_no',
            'SalesChannel': 'sales_channel',
            'Customer_VAT_Status': 'vat_status',
            'UNQTRN': 'unique_transaction',
            'No of Trans 2024': 'num_transactions'
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from Excel file.
        
        Returns:
            DataFrame with raw data.
            
        Raises:
            FileNotFoundError: If the raw data file is not found.
            ValueError: If the data cannot be loaded.
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        try:
            logger.info(f"Loading raw data from {self.raw_data_path}")
            df = pd.read_excel(self.raw_data_path, engine='openpyxl')
            logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading raw data: {e}")
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names using the predefined mapping.
        
        Args:
            df: DataFrame with raw column names.
            
        Returns:
            DataFrame with standardized column names.
        """
        df = df.copy()
        
        # Apply column mapping
        for old_name, new_name in self.column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        logger.info(f"Standardized columns: {df.columns.tolist()}")
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse invoice dates with custom Jarir date format handling.
        
        Args:
            df: DataFrame with invoice_date column.
            
        Returns:
            DataFrame with parsed dates.
        """
        df = df.copy()
        
        if 'invoice_date' not in df.columns:
            logger.warning("No invoice_date column found, skipping date parsing")
            return df
        
        logger.info("Parsing invoice dates")
        df['invoice_date'] = df['invoice_date'].apply(self._parse_single_date)
        
        # Validate date parsing
        date_valid_ratio = 1.0 - df['invoice_date'].isna().mean()
        logger.info(f"Date parsing success rate: {date_valid_ratio:.2%}")
        
        if date_valid_ratio == 0.0:
            raise ValueError(
                "All dates failed to parse. Please check source date formats."
            )
        
        return df
    
    def _parse_single_date(self, date_value: Any) -> pd.Timestamp:
        """
        Parse a single date value with multiple format support.
        
        Args:
            date_value: Raw date value from the dataset.
            
        Returns:
            Parsed timestamp or NaT if parsing fails.
        """
        if pd.isna(date_value):
            return pd.NaT
        
        # Handle Excel serial numbers
        if isinstance(date_value, (int, float)):
            try:
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(date_value))
            except Exception:
                return pd.NaT
        
        if isinstance(date_value, str):
            date_str = date_value.strip()
            
            # Handle Jarir custom format 'Mon-DD' or 'Mon DD'
            month_pattern = re.compile(
                r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\-\s](\d{1,2})$",
                re.IGNORECASE
            )
            match = month_pattern.match(date_str)
            
            if match:
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month_str, day_str = match.group(1).title(), match.group(2)
                try:
                    return pd.Timestamp(2024, month_map[month_str], int(day_str))
                except Exception:
                    return pd.NaT
            
            # Try generic pandas parsing
            for dayfirst in [True, False]:
                try:
                    return pd.to_datetime(date_str, dayfirst=dayfirst, errors='raise')
                except Exception:
                    continue
        
        return pd.NaT
    
    def compute_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute unit prices from line amounts and quantities.
        
        Args:
            df: DataFrame with quantity and line_amount columns.
            
        Returns:
            DataFrame with computed price column.
        """
        df = df.copy()
        
        if 'quantity' not in df.columns or 'line_amount' not in df.columns:
            logger.warning("Missing quantity or line_amount columns for price computation")
            return df
        
        logger.info("Computing unit prices")
        
        # Convert to numeric, handling errors
        quantity = pd.to_numeric(df['quantity'], errors='coerce')
        line_amount = pd.to_numeric(df['line_amount'], errors='coerce')
        
        # Compute price with division by zero handling
        with np.errstate(divide='ignore', invalid='ignore'):
            df['price'] = line_amount / quantity
        
        # Set non-finite prices to NaN
        df.loc[~np.isfinite(df['price']), 'price'] = np.nan
        
        logger.info(f"Computed prices for {df['price'].notna().sum()} rows")
        return df


def load_processed_sequences(
    data_dir: Union[str, Path],
    splits: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load processed sequence data for train/validation/test splits.
    
    Args:
        data_dir: Directory containing processed sequence files.
        splits: List of splits to load. Defaults to ['train', 'val', 'test'].
        
    Returns:
        Dictionary mapping split names to DataFrames.
        
    Raises:
        FileNotFoundError: If required sequence files are missing.
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    data_dir = Path(data_dir)
    sequences = {}
    
    for split in splits:
        file_path = data_dir / f'sequences_{split}.parquet'
        if not file_path.exists():
            raise FileNotFoundError(f"Sequence file not found: {file_path}")
        
        logger.info(f"Loading {split} sequences from {file_path}")
        sequences[split] = read_parquet(file_path)
        logger.info(f"Loaded {len(sequences[split])} {split} sequences")
    
    return sequences


def load_id_mappings(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load item and customer ID mapping tables.
    
    Args:
        data_dir: Directory containing ID mapping files.
        
    Returns:
        Tuple of (item_map, customer_map) DataFrames.
        
    Raises:
        FileNotFoundError: If mapping files are missing.
    """
    data_dir = Path(data_dir)
    
    item_map_path = data_dir / 'item_id_map.parquet'
    customer_map_path = data_dir / 'customer_id_map.parquet'
    
    if not item_map_path.exists():
        raise FileNotFoundError(f"Item mapping file not found: {item_map_path}")
    
    if not customer_map_path.exists():
        raise FileNotFoundError(f"Customer mapping file not found: {customer_map_path}")
    
    logger.info("Loading ID mappings")
    item_map = read_parquet(item_map_path)
    customer_map = read_parquet(customer_map_path)
    
    logger.info(f"Loaded {len(item_map)} items, {len(customer_map)} customers")
    
    return item_map, customer_map


def parse_history_string(history_str: str) -> List[int]:
    """
    Parse history string into list of item indices.
    
    Args:
        history_str: Space-separated string of item indices.
        
    Returns:
        List of item indices.
    """
    if not isinstance(history_str, str) or not history_str.strip():
        return []
    
    try:
        return [int(x) for x in history_str.strip().split()]
    except ValueError:
        logger.warning(f"Invalid history string: {history_str}")
        return []


def create_interaction_matrix(
    sequences: pd.DataFrame,
    n_users: int,
    n_items: int,
    history_weight: float = 0.5,
    positive_weight: float = 1.0,
) -> 'scipy.sparse.csr_matrix':
    """
    Create sparse user-item interaction matrix from sequences.
    
    Args:
        sequences: DataFrame with user_idx, pos_item_idx, and history_idx columns.
        n_users: Number of users in the system.
        n_items: Number of items in the system.
        history_weight: Weight for history interactions.
        positive_weight: Weight for positive interactions.
        
    Returns:
        Sparse CSR matrix of user-item interactions.
    """
    from scipy.sparse import csr_matrix
    
    rows, cols, vals = [], [], []
    
    for _, row in sequences.iterrows():
        user_idx = int(row['user_idx'])
        pos_item_idx = int(row['pos_item_idx'])
        
        # Add positive interaction
        rows.append(user_idx)
        cols.append(pos_item_idx)
        vals.append(positive_weight)
        
        # Add history interactions
        if pd.notna(row['history_idx']) and row['history_idx']:
            history = parse_history_string(str(row['history_idx']))
            for item_idx in history:
                rows.append(user_idx)
                cols.append(item_idx)
                vals.append(history_weight)
    
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

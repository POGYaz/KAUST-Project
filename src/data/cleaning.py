"""
Data cleaning utilities for the recommendation system.

This module provides functions for cleaning and preprocessing raw transaction
data, including outlier detection, text normalization, and data validation.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for transaction data.
    
    Handles various cleaning operations including missing value handling,
    outlier detection, text normalization, and duplicate removal.
    """
    
    def __init__(
        self,
        keep_only_positive_qty: bool = True,
        keep_only_positive_price: bool = True,
        drop_returns_by_prefix: bool = True,
        drop_duplicate_rows: bool = True,
        handle_outliers: bool = True,
        winsorize_instead_of_drop: bool = False,
        iqr_multiplier: float = 3.0,
        min_events_per_user: int = 1,
        min_purchases_per_item: int = 1,
    ):
        """
        Initialize the data cleaner with configuration options.
        
        Args:
            keep_only_positive_qty: Whether to keep only positive quantities.
            keep_only_positive_price: Whether to keep only positive prices.
            drop_returns_by_prefix: Whether to drop returns by transaction prefix.
            drop_duplicate_rows: Whether to drop exact duplicate rows.
            handle_outliers: Whether to handle outliers using IQR method.
            winsorize_instead_of_drop: Whether to winsorize outliers instead of dropping.
            iqr_multiplier: Multiplier for IQR-based outlier detection.
            min_events_per_user: Minimum events per user to keep.
            min_purchases_per_item: Minimum purchases per item to keep.
        """
        self.keep_only_positive_qty = keep_only_positive_qty
        self.keep_only_positive_price = keep_only_positive_price
        self.drop_returns_by_prefix = drop_returns_by_prefix
        self.drop_duplicate_rows = drop_duplicate_rows
        self.handle_outliers = handle_outliers
        self.winsorize_instead_of_drop = winsorize_instead_of_drop
        self.iqr_multiplier = iqr_multiplier
        self.min_events_per_user = min_events_per_user
        self.min_purchases_per_item = min_purchases_per_item
        
        # Non-product patterns for filtering
        self.non_product_patterns = [
            r'POSTAGE', r'SHIPPING', r'CARRIAGE', r'DELIVERY',
            r'BANK CHARGES', r'AMAZON', r'DOTCOM', r'PACKING',
            r'ADJUST', r'DISCOUNT', r'SAMPLE', r'SAMPLES',
            r'CHECK', r'TEST', r'MANUAL', r'FEE', r'CHARGE'
        ]
        
        # Quality metrics tracking
        self.quality_metrics: Dict[str, int] = {}
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply complete cleaning pipeline to the data.
        
        Args:
            df: Raw DataFrame to clean.
            
        Returns:
            Tuple of (cleaned_df, quality_metrics).
        """
        logger.info(f"Starting data cleaning pipeline with {len(df)} rows")
        
        # Reset quality metrics
        self.quality_metrics = {}
        
        # Step 1: Basic validity filters
        df = self._apply_basic_filters(df)
        
        # Step 2: Text normalization and non-product removal
        df = self._normalize_and_filter_text(df)
        
        # Step 3: Aggregate duplicate lines
        df = self._aggregate_duplicates(df)
        
        # Step 4: Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df)
        
        # Step 5: Coverage filters
        df = self._apply_coverage_filters(df)
        
        logger.info(f"Cleaning complete: {len(df)} rows remaining")
        return df, self.quality_metrics.copy()
    
    def _apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic validity filters to the data."""
        df = df.copy()
        
        # Filter 1: Missing core fields
        before = len(df)
        core_columns = ['invoice_date', 'stock_code', 'description', 'country', 'customer_id']
        available_cores = [col for col in core_columns if col in df.columns]
        
        if available_cores:
            core_mask = df[available_cores].notna().all(axis=1)
            df = df[core_mask]
            self.quality_metrics['drop_missing_core'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_missing_core']} rows with missing core fields")
        
        # Filter 2: Returns by transaction prefix
        if self.drop_returns_by_prefix and 'unique_transaction' in df.columns:
            before = len(df)
            returns_mask = ~df['unique_transaction'].astype(str).str.startswith('C', na=False)
            df = df[returns_mask]
            self.quality_metrics['drop_returns'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_returns']} return transactions")
        
        # Filter 3: Positive quantities
        if self.keep_only_positive_qty and 'quantity' in df.columns:
            before = len(df)
            positive_qty_mask = pd.to_numeric(df['quantity'], errors='coerce') > 0
            df = df[positive_qty_mask]
            self.quality_metrics['drop_nonpositive_qty'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_nonpositive_qty']} non-positive quantities")
        
        # Filter 4: Positive prices
        if self.keep_only_positive_price and 'price' in df.columns:
            before = len(df)
            positive_price_mask = pd.to_numeric(df['price'], errors='coerce') > 0
            df = df[positive_price_mask]
            self.quality_metrics['drop_nonpositive_price'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_nonpositive_price']} non-positive prices")
        
        # Filter 5: Duplicate rows
        if self.drop_duplicate_rows:
            before = len(df)
            df = df.drop_duplicates()
            self.quality_metrics['drop_duplicates'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_duplicates']} duplicate rows")
        
        return df
    
    def _normalize_and_filter_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text and filter non-product items."""
        df = df.copy()
        
        # Normalize text fields
        if 'description' in df.columns:
            df['description'] = df['description'].apply(self._normalize_text)
        
        if 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str).str.upper()
        
        # Filter non-product patterns
        if 'description' in df.columns:
            before = len(df)
            pattern = re.compile('|'.join(self.non_product_patterns), flags=re.IGNORECASE)
            non_product_mask = ~df['description'].str.contains(pattern, na=False)
            df = df[non_product_mask]
            self.quality_metrics['drop_non_product_patterns'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_non_product_patterns']} non-product items")
        
        return df
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by handling Unicode, whitespace, and case.
        
        Args:
            text: Raw text string.
            
        Returns:
            Normalized text string.
        """
        if not isinstance(text, str):
            text = '' if pd.isna(text) else str(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text).strip()
        
        # Whitespace normalization
        text = re.sub(r'\\s+', ' ', text)
        
        return text.upper()
    
    def _aggregate_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate duplicate transaction lines."""
        df = df.copy()
        
        # Check if we have the necessary columns for aggregation
        groupby_cols = ['customer_id', 'stock_code', 'invoice_date']
        available_groupby = [col for col in groupby_cols if col in df.columns]
        
        if len(available_groupby) < 2:
            logger.warning("Insufficient columns for duplicate aggregation, skipping")
            return df
        
        before = len(df)
        
        # Recompute line_amount from price and quantity if possible
        if all(col in df.columns for col in ['price', 'quantity']):
            valid_price_mask = df['price'].notna()
            df.loc[valid_price_mask, 'line_amount'] = (
                df.loc[valid_price_mask, 'quantity'] * df.loc[valid_price_mask, 'price']
            )
        
        # Define aggregation rules
        agg_rules = {}
        
        # Numeric columns to sum
        numeric_sum_cols = ['quantity', 'line_amount']
        for col in numeric_sum_cols:
            if col in df.columns:
                agg_rules[col] = 'sum'
        
        # Categorical columns to take first value
        categorical_first_cols = [
            'description', 'country', 'brand', 'category', 'vendor'
        ]
        for col in categorical_first_cols:
            if col in df.columns:
                agg_rules[col] = 'first'
        
        if agg_rules:
            # Perform aggregation
            df = df.groupby(available_groupby, as_index=False).agg(agg_rules)
            
            # Recompute weighted price after aggregation
            if all(col in df.columns for col in ['line_amount', 'quantity']):
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['price'] = df['line_amount'] / df['quantity']
                    df.loc[~np.isfinite(df['price']), 'price'] = np.nan
        
        self.quality_metrics['aggregated_duplicate_lines'] = before - len(df)
        logger.info(f"Aggregated {self.quality_metrics['aggregated_duplicate_lines']} duplicate lines")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method on log scale."""
        df = df.copy()
        
        outlier_columns = ['price', 'quantity', 'line_amount']
        available_outlier_cols = [col for col in outlier_columns if col in df.columns]
        
        if not available_outlier_cols:
            logger.warning("No columns available for outlier handling")
            return df
        
        before = len(df)
        
        # Compute IQR bounds for each column
        bounds = {}
        for col in available_outlier_cols:
            bounds[col] = self._compute_iqr_bounds_log(df[col])
        
        # Create outlier mask
        outlier_mask = pd.Series(True, index=df.index)
        
        for col, (lower, upper) in bounds.items():
            col_mask = df[col].between(lower, upper, inclusive='both')
            outlier_mask &= col_mask
        
        if self.winsorize_instead_of_drop:
            # Winsorize outliers
            for col, (lower, upper) in bounds.items():
                df[col] = df[col].clip(lower, upper)
            
            # Recompute line_amount if price or quantity were winsorized
            if 'price' in bounds and 'quantity' in bounds and 'line_amount' in df.columns:
                df['line_amount'] = df['quantity'] * df['price']
            
            removed = 0
        else:
            # Drop outliers
            df = df[outlier_mask]
            removed = before - len(df)
        
        self.quality_metrics['drop_outliers'] = removed
        logger.info(f"Handled {before - len(df)} outlier rows")
        
        return df
    
    def _compute_iqr_bounds_log(
        self,
        series: pd.Series,
        log_epsilon: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Compute IQR bounds on log scale for outlier detection.
        
        Args:
            series: Numeric series to compute bounds for.
            log_epsilon: Small value to add before log transformation.
            
        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        # Log transform with epsilon to handle zeros
        log_series = np.log(series.clip(lower=log_epsilon)).dropna()
        
        if len(log_series) == 0:
            return 0.0, np.inf
        
        # Compute IQR on log scale
        q1, q3 = np.percentile(log_series, [25, 75])
        iqr = q3 - q1
        
        # Compute bounds and transform back to original scale
        log_lower = q1 - self.iqr_multiplier * iqr
        log_upper = q3 + self.iqr_multiplier * iqr
        
        return np.exp(log_lower), np.exp(log_upper)
    
    def _apply_coverage_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply coverage filters for rare users and items."""
        df = df.copy()
        
        # Filter rare items
        if self.min_purchases_per_item > 1 and 'stock_code' in df.columns:
            before = len(df)
            item_counts = df['stock_code'].value_counts()
            frequent_items = item_counts[item_counts >= self.min_purchases_per_item].index
            df = df[df['stock_code'].isin(frequent_items)]
            self.quality_metrics['drop_rare_items'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_rare_items']} rows with rare items")
        
        # Filter rare users
        if self.min_events_per_user > 1 and 'customer_id' in df.columns:
            before = len(df)
            user_counts = df['customer_id'].value_counts()
            frequent_users = user_counts[user_counts >= self.min_events_per_user].index
            df = df[df['customer_id'].isin(frequent_users)]
            self.quality_metrics['drop_rare_users'] = before - len(df)
            logger.info(f"Dropped {self.quality_metrics['drop_rare_users']} rows with rare users")
        
        return df


def create_clean_tables(
    interactions_df: pd.DataFrame,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create clean interaction, item catalog, and customer tables.
    
    Args:
        interactions_df: Cleaned interactions DataFrame.
        output_dir: Directory to save the clean tables.
        
    Returns:
        Tuple of (interactions, item_catalog, customer_table).
    """
    from pathlib import Path
    from ..utils.io import write_parquet
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating clean tables")
    
    # Sort interactions by customer and date
    interactions = interactions_df[
        ['customer_id', 'invoice_date', 'stock_code', 'description',
         'quantity', 'price', 'line_amount', 'country', 'brand', 'category', 'vendor']
    ].sort_values(['customer_id', 'invoice_date']).reset_index(drop=True)
    
    # Create item catalog (use named aggregation to avoid MultiIndex issues)
    item_catalog = (
        interactions.groupby('stock_code', as_index=False)
        .agg(
            description=('description', lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            price_median=('price', 'median'),
            price_mean=('price', 'mean'),
            popularity=('stock_code', 'size'),
            brand=('brand', lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            category=('category', lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            vendor=('vendor', lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        )
    )
    item_catalog = item_catalog.sort_values('popularity', ascending=False)
    
    # Create customer table
    customer_table = (
        interactions.groupby('customer_id', as_index=False)
        .agg({
            'invoice_date': ['min', 'max', 'nunique'],
            'stock_code': 'size',
            'country': lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
            'line_amount': 'sum',
        })
    )
    
    # Flatten column names
    customer_table.columns = [
        'customer_id', 'first_date', 'last_date',
        'n_events', 'n_lines', 'country_mode', 'total_spent'
    ]
    
    # Save tables
    write_parquet(interactions, output_dir / 'interactions_clean.parquet')
    write_parquet(item_catalog, output_dir / 'items_clean.parquet')
    write_parquet(customer_table, output_dir / 'customers_clean.parquet')
    
    logger.info(f"Saved clean tables to {output_dir}")
    logger.info(f"Interactions: {len(interactions)} rows")
    logger.info(f"Items: {len(item_catalog)} items")
    logger.info(f"Customers: {len(customer_table)} customers")
    
    return interactions, item_catalog, customer_table

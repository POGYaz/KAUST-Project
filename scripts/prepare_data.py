#!/usr/bin/env python3
"""
Data preparation script for the Jarir recommendation system.

This script loads raw data, applies cleaning and preprocessing steps,
and generates clean interaction tables and user sequences.

Example usage:
    python scripts/prepare_data.py --config configs/data.yaml
    python scripts/prepare_data.py --config configs/data.yaml --output-dir data/processed/custom
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaning import DataCleaner, create_clean_tables
from src.data.dataset import JarirDatasetLoader
from src.data.features import SequenceBuilder, create_id_mappings
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare data for the Jarir recommendation system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data configuration YAML file"
    )
    
    parser.add_argument(
        "--raw-data-path",
        type=str,
        help="Path to raw data file (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for processed data (overrides config)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (overrides config)"
    )
    
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip sequence generation step"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing output files"
    )
    
    return parser.parse_args()


def main():
    """Main data preparation pipeline."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level, rich_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preparation pipeline")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override config with command line arguments
        if args.raw_data_path:
            config['input']['raw_data_path'] = args.raw_data_path
        if args.output_dir:
            config['input']['output_dir'] = args.output_dir
        if args.seed:
            config['random_seed'] = args.seed
        
        # Set random seed
        seed = config.get('random_seed', 42)
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")
        
        # Initialize paths
        raw_data_path = Path(config['input']['raw_data_path'])
        output_dir = Path(config['input']['output_dir'])
        
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load and standardize raw data
        logger.info("Step 1: Loading and standardizing raw data")
        
        dataset_loader = JarirDatasetLoader(raw_data_path, output_dir)
        raw_data = dataset_loader.load_raw_data()
        
        # Standardize columns and parse dates
        data = dataset_loader.standardize_columns(raw_data)
        data = dataset_loader.parse_dates(data)
        data = dataset_loader.compute_prices(data)
        
        logger.info(f"Loaded and standardized {len(data)} rows")
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data")
        
        cleaner_config = config.get('cleaning', {})
        data_cleaner = DataCleaner(
            keep_only_positive_qty=cleaner_config.get('keep_only_positive_qty', True),
            keep_only_positive_price=cleaner_config.get('keep_only_positive_price', True),
            drop_returns_by_prefix=cleaner_config.get('drop_returns_by_prefix', True),
            drop_duplicate_rows=cleaner_config.get('drop_duplicate_rows', True),
            handle_outliers=cleaner_config.get('handle_outliers', True),
            winsorize_instead_of_drop=cleaner_config.get('winsorize_instead_of_drop', False),
            iqr_multiplier=cleaner_config.get('iqr_multiplier', 3.0),
            min_events_per_user=cleaner_config.get('min_events_per_user', 1),
            min_purchases_per_item=cleaner_config.get('min_purchases_per_item', 1),
        )
        
        clean_data, quality_metrics = data_cleaner.clean_data(data)
        logger.info(f"Cleaned data: {len(clean_data)} rows remaining")
        
        # Step 3: Create clean tables
        logger.info("Step 3: Creating clean tables")
        
        if not args.dry_run:
            interactions, item_catalog, customer_table = create_clean_tables(
                clean_data, output_dir
            )
            
            # Save quality report
            quality_report = {
                'input': {
                    'raw_data_path': str(raw_data_path),
                    'raw_rows': len(raw_data),
                },
                'output': {
                    'clean_rows': len(clean_data),
                    'interactions': len(interactions),
                    'items': len(item_catalog),
                    'customers': len(customer_table),
                },
                'quality_metrics': quality_metrics,
                'config': config,
            }
            
            write_json(quality_report, output_dir / 'quality_report.json')
            logger.info(f"Saved quality report to {output_dir / 'quality_report.json'}")
        
        # Step 4: Create ID mappings
        logger.info("Step 4: Creating ID mappings")
        
        if not args.dry_run:
            item_map, customer_map = create_id_mappings(clean_data, output_dir)
            logger.info(f"Created {len(item_map)} item mappings and {len(customer_map)} customer mappings")
        
        # Step 5: Generate sequences (optional)
        if not args.skip_sequences:
            logger.info("Step 5: Generating user sequences")
            
            if not args.dry_run:
                sequence_config = config.get('sequences', {})
                sequence_builder = SequenceBuilder(
                    max_history_length=sequence_config.get('max_history_length', 15),
                    min_history_length=sequence_config.get('min_history_length', 2),
                    train_split_quantile=sequence_config.get('train_split_quantile', 0.80),
                    val_split_quantile=sequence_config.get('val_split_quantile', 0.90),
                    random_negatives=sequence_config.get('add_random_negatives', False),
                    n_negatives_train=sequence_config.get('n_negatives_train', 50),
                    n_negatives_val=sequence_config.get('n_negatives_val', 100),
                    random_seed=seed,
                )
                
                sequences = sequence_builder.build_sequences(
                    clean_data, item_map, customer_map, output_dir
                )
                
                total_sequences = sum(len(seq_df) for seq_df in sequences.values())
                logger.info(f"Generated {total_sequences} total sequences")
        
        # Validation
        if config.get('validation', {}).get('enabled', True):
            logger.info("Step 6: Validating processed data")
            
            validation_config = config.get('validation', {})
            
            # Basic size checks
            min_interactions = validation_config.get('min_interactions', 1000)
            min_users = validation_config.get('min_users', 100)
            min_items = validation_config.get('min_items', 100)
            
            if len(clean_data) < min_interactions:
                logger.warning(f"Dataset has only {len(clean_data)} interactions (minimum: {min_interactions})")
            
            if not args.dry_run:
                if len(customer_map) < min_users:
                    logger.warning(f"Dataset has only {len(customer_map)} users (minimum: {min_users})")
                
                if len(item_map) < min_items:
                    logger.warning(f"Dataset has only {len(item_map)} items (minimum: {min_items})")
        
        logger.info("Data preparation completed successfully!")
        
        if args.dry_run:
            logger.info("Dry run completed - no files were written")
        else:
            logger.info(f"Processed data saved to: {output_dir}")
            
            # Print summary
            print("\n" + "="*50)
            print("DATA PREPARATION SUMMARY")
            print("="*50)
            print(f"Raw data rows: {len(raw_data):,}")
            print(f"Clean data rows: {len(clean_data):,}")
            if not args.dry_run:
                print(f"Items: {len(item_map):,}")
                print(f"Customers: {len(customer_map):,}")
                if not args.skip_sequences:
                    print(f"Total sequences: {total_sequences:,}")
            print(f"Output directory: {output_dir}")
            print("="*50)
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Two-Tower retriever training script for the Jarir recommendation system.

This script trains a Two-Tower neural retrieval model using processed
sequence data and saves the trained model and embeddings.

Example usage:
    python scripts/train_retriever.py --config configs/retriever.yaml
    python scripts/train_retriever.py --config configs/retriever.yaml --epochs 10 --batch-size 256
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_processed_sequences, load_id_mappings
from src.models.retriever.two_tower import TwoTowerModel, SequenceDataset, collate_sequences
from src.models.retriever.losses import create_loss_function
from src.training.trainer import RetrievalTrainer, setup_training
from src.training.callbacks import EarlyStopping, ModelCheckpoint, MetricsLogger
from src.utils.config import load_config
from src.utils.io import write_json, write_numpy
from src.utils.logging import setup_logging
from src.utils.seed import set_seed, get_device


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Two-Tower retrieval model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to retriever configuration YAML file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing processed data (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model and embeddings (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (overrides config)"
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
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation without training"
    )
    
    parser.add_argument(
        "--export-embeddings",
        action="store_true",
        help="Export user and item embeddings after training"
    )
    
    return parser.parse_args()


def load_data(config, data_dir):
    """Load and prepare training data."""
    logger = logging.getLogger(__name__)
    
    # Load sequences and mappings
    sequences = load_processed_sequences(data_dir, splits=['train', 'val'])
    item_map, customer_map = load_id_mappings(data_dir)
    
    logger.info(f"Loaded {len(sequences['train'])} training sequences")
    logger.info(f"Loaded {len(sequences['val'])} validation sequences")
    logger.info(f"Dataset: {len(customer_map)} users, {len(item_map)} items")
    
    # Notebook parity: dev split from train by ts quantile
    dev_q = float(config.get('data', {}).get('dev_split_quantile', 0.90))
    seq_train = sequences['train']
    if 'ts' in seq_train.columns and len(seq_train) > 0:
        cut_ts = seq_train['ts'].quantile(dev_q)
        seq_train_tr = seq_train[seq_train['ts'] < cut_ts].reset_index(drop=True)
        seq_train_dev = seq_train[seq_train['ts'] >= cut_ts].reset_index(drop=True)
    else:
        seq_train_tr = seq_train
        seq_train_dev = sequences['val']

    # Create datasets
    data_config = config.get('data', {})
    train_dataset = SequenceDataset(
        seq_train_tr,
        n_items=len(item_map),
        n_negatives=data_config.get('k_negatives', 50),
        random_seed=config.get('random_seed', 42)
    )
    dev_dataset = SequenceDataset(
        seq_train_dev,
        n_items=len(item_map),
        n_negatives=data_config.get('k_negatives', 50),
        random_seed=config.get('random_seed', 42)
    )
    
    return train_dataset, dev_dataset, len(customer_map), len(item_map)


def create_data_loaders(train_dataset, val_dataset, config):
    """Create data loaders for training and validation."""
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 512),
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=data_config.get('persistent_workers', True),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 512),
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=data_config.get('persistent_workers', True),
    )
    
    return train_loader, val_loader


def create_model(config, n_users, n_items, device):
    """Create and initialize the Two-Tower model."""
    model_config = config.get('model', {})
    
    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        d_model=model_config.get('d_model', 256),
        dropout=model_config.get('dropout', 0.2),
        n_blocks=model_config.get('n_blocks', 2),
        embedding_init_std=model_config.get('embedding_init_std', 0.1),
    )
    
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.getLogger(__name__).info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    return model


def export_embeddings(model, output_dir, device):
    """Export trained user and item embeddings."""
    logger = logging.getLogger(__name__)
    
    logger.info("Exporting embeddings...")
    
    with torch.no_grad():
        # Export user embeddings
        user_embeddings = model.get_all_user_embeddings(device)
        user_embeddings_np = user_embeddings.cpu().numpy()
        
        # Export item embeddings
        item_embeddings = model.get_all_item_embeddings(device)
        item_embeddings_np = item_embeddings.cpu().numpy()
    
    # Save embeddings
    user_emb_path = output_dir / 'user_embeddings.npy'
    item_emb_path = output_dir / 'item_embeddings.npy'
    
    write_numpy(user_embeddings_np, user_emb_path)
    write_numpy(item_embeddings_np, item_emb_path)
    
    logger.info(f"Saved user embeddings: {user_emb_path} {user_embeddings_np.shape}")
    logger.info(f"Saved item embeddings: {item_emb_path} {item_embeddings_np.shape}")


def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level, rich_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Two-Tower retriever training")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override config with command line arguments
        if args.data_dir:
            config['data_dir'] = args.data_dir
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['optimizer']['params']['lr'] = args.learning_rate
        if args.device:
            config['hardware']['device'] = args.device
        if args.seed:
            config['random_seed'] = args.seed
        
        # Set random seed
        seed = config.get('random_seed', 42)
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")
        
        # Determine device
        device_config = config.get('hardware', {}).get('device', 'auto')
        if device_config == 'auto':
            device = get_device()
        else:
            device = device_config
        
        logger.info(f"Using device: {device}")
        
        # Initialize paths
        data_dir = Path(config.get('data_dir', 'data/processed/jarir'))
        output_dir = Path(config.get('output_dir', 'models/retriever'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading training data...")
        train_dataset, dev_dataset, n_users, n_items = load_data(config, data_dir)
        
        # Create data loaders (dev used for validation/early stopping)
        train_loader, val_loader = create_data_loaders(train_dataset, dev_dataset, config)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, n_users, n_items, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint)
        
        # Setup training components
        optimizer, scheduler, loss_function = setup_training(model, config, device)
        
        # Create trainer
        training_config = config.get('training', {})
        trainer = RetrievalTrainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            scheduler=scheduler,
            gradient_clip_value=training_config.get('gradient_clip_value'),
            accumulation_steps=training_config.get('accumulation_steps', 1),
            mixed_precision=training_config.get('mixed_precision', False),
        )
        
        # Add callbacks (use config filepath and monitor)
        checkpoint_config = config.get('checkpointing', {})
        if checkpoint_config.get('enabled', True):
            ckpt_path = checkpoint_config.get('filepath', str(Path('models/retriever') / 'best_model.pth'))
            checkpoint_callback = ModelCheckpoint(
                filepath=Path(ckpt_path),
                monitor=checkpoint_config.get('monitor', 'val_recall@10'),
                mode=checkpoint_config.get('mode', 'max'),
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_weights_only=checkpoint_config.get('save_weights_only', True),
            )
            checkpoint_callback.set_model(model)
            trainer.add_callback(checkpoint_callback)
        
        early_stopping_config = training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            early_stopping = EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_recall@10'),
                patience=early_stopping_config.get('patience', 5),
                mode=early_stopping_config.get('mode', 'max'),
                min_delta=early_stopping_config.get('min_delta', 0.0001),
            )
            trainer.add_callback(early_stopping)
        
        metrics_logger = MetricsLogger(
            save_path=output_dir / 'training_metrics.json',
            save_frequency=5,
        )
        trainer.add_callback(metrics_logger)
        
        if not args.eval_only:
            # Train model
            logger.info("Starting training...")
            
            epochs = training_config.get('epochs', 50)
            history = trainer.fit(
                train_loader=train_loader,
                validation_loader=val_loader,
                epochs=epochs,
                save_dir=output_dir,
            )
            
            logger.info("Training completed!")
            
            # Save final model
            final_model_path = output_dir / 'final_model.pth'
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        
        # Export embeddings (use export_path from config for notebook parity)
        if args.export_embeddings or config.get('embeddings', {}).get('export_embeddings', True):
            export_dir = Path(config.get('embeddings', {}).get('export_path', str(output_dir)))
            export_dir.mkdir(parents=True, exist_ok=True)
            export_embeddings(model, export_dir, device)
        
        # Save configuration
        config_path = output_dir / 'config.json'
        write_json(config, config_path)
        logger.info(f"Saved configuration to {config_path}")
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model: Two-Tower ({n_users:,} users, {n_items:,} items)")
        print(f"Training sequences: {len(train_dataset):,}")
        print(f"Dev sequences: {len(dev_dataset):,}")
        if not args.eval_only:
            print(f"Epochs trained: {epochs}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")
        print("="*50)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation script for the Jarir recommendation system.

This script runs comprehensive evaluation of trained models including
baselines, retrieval, and reranking performance with standard metrics.

Example usage:
    python scripts/evaluate.py --data-dir data/processed/jarir --model-dir models
    python scripts/evaluate.py --config configs/evaluation.yaml --k-values 5 10 20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset import load_processed_sequences, load_id_mappings, create_interaction_matrix
from src.evaluation.eval_pipeline import BaselineEvaluator, run_full_evaluation
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config
from src.utils.io import read_json, read_numpy, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed data"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation metrics"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Data splits to evaluate on"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["baselines", "retrieval", "reranking"],
        choices=["baselines", "retrieval", "reranking"],
        help="Models to evaluate"
    )
    
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate baseline models"
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
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for evaluation (for faster testing)"
    )
    
    return parser.parse_args()


def evaluate_baselines(data_dir, k_values, splits, sample_size=None):
    """Evaluate baseline recommendation models."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating baseline models")
    
    # Load data
    sequences = load_processed_sequences(data_dir, splits=['train'] + splits)
    item_map, customer_map = load_id_mappings(data_dir)
    
    n_users, n_items = len(customer_map), len(item_map)
    
    # Create interaction matrices
    train_matrix = create_interaction_matrix(sequences['train'], n_users, n_items)
    
    # Try to load denser interaction matrix for KNN
    try:
        interactions_path = Path(data_dir) / 'interactions_clean.parquet'
        if interactions_path.exists():
            interactions = pd.read_parquet(interactions_path)
            interactions = interactions.merge(item_map, on='stock_code', how='inner')
            interactions = interactions.merge(customer_map, on='customer_id', how='inner')
            
            # Filter to training period
            if 'ts' in sequences['train'].columns:
                cutoff = pd.to_datetime(sequences['train']['ts'].max())
                if 'invoice_date' in interactions.columns:
                    interactions = interactions[interactions['invoice_date'] <= cutoff]
            
            rows = interactions['user_idx'].astype(int).values
            cols = interactions['item_idx'].astype(int).values
            vals = np.ones_like(rows, dtype=np.float32)
            
            from scipy.sparse import csr_matrix
            knn_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
            logger.info(f"Using denser KNN matrix: {knn_matrix.nnz:,} interactions")
        else:
            knn_matrix = train_matrix
    except Exception as e:
        logger.warning(f"Could not load denser interaction matrix: {e}")
        knn_matrix = train_matrix
    
    # Initialize baseline evaluator
    evaluator = BaselineEvaluator(k_values=k_values)
    
    # Fit baseline models
    baseline_models = evaluator.fit_baselines(train_matrix, knn_matrix)
    
    # Evaluate on each split
    results = {}
    
    for split in splits:
        if split not in sequences:
            logger.warning(f"Split '{split}' not found in sequences")
            continue
        
        logger.info(f"Evaluating baselines on {split} split")
        
        # Prepare test interactions
        test_sequences = sequences[split]
        if sample_size and len(test_sequences) > sample_size:
            test_sequences = test_sequences.sample(n=sample_size, random_state=42)
        
        test_interactions = [
            (int(row['user_idx']), int(row['pos_item_idx']))
            for _, row in test_sequences.iterrows()
        ]
        
        # Evaluate baselines
        split_results = evaluator.evaluate_baselines(
            test_interactions, train_matrix, n_items
        )
        
        results[split] = split_results
    
    return results


def evaluate_retrieval_model(data_dir, model_dir, k_values, splits, sample_size=None):
    """Evaluate retrieval model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating retrieval model")
    
    # Check if embeddings exist
    user_emb_path = Path(model_dir) / 'user_embeddings.npy'
    item_emb_path = Path(model_dir) / 'item_embeddings.npy'
    
    if not (user_emb_path.exists() and item_emb_path.exists()):
        logger.warning("Retrieval model embeddings not found, skipping retrieval evaluation")
        return {}
    
    # Load embeddings
    user_embeddings = read_numpy(user_emb_path)
    item_embeddings = read_numpy(item_emb_path)
    
    logger.info(f"Loaded embeddings: users {user_embeddings.shape}, items {item_embeddings.shape}")
    
    # Load data
    sequences = load_processed_sequences(data_dir, splits=splits)
    
    # Evaluate on each split
    results = {}
    
    for split in splits:
        if split not in sequences:
            continue
        
        logger.info(f"Evaluating retrieval on {split} split")
        
        test_sequences = sequences[split]
        if sample_size and len(test_sequences) > sample_size:
            test_sequences = test_sequences.sample(n=sample_size, random_state=42)
        
        # Prepare test interactions
        test_interactions = [
            (int(row['user_idx']), int(row['pos_item_idx']))
            for _, row in test_sequences.iterrows()
        ]
        
        # Evaluate retrieval using embeddings
        from src.evaluation.eval_pipeline import RetrievalEvaluator
        
        retrieval_evaluator = RetrievalEvaluator(k_values=k_values)
        split_results = retrieval_evaluator.evaluate_retrieval(
            user_embeddings, item_embeddings, test_interactions, len(item_embeddings)
        )
        
        results[split] = split_results
    
    return results


def evaluate_reranking_model(data_dir, model_dir, k_values, splits, sample_size=None):
    """Evaluate reranking model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating reranking model")
    
    # Check if reranker model exists
    ranker_path = Path(model_dir) / 'ranker_best.pt'
    
    if not ranker_path.exists():
        logger.warning("Reranker model not found, skipping reranking evaluation")
        return {}
    
    # Check if candidate files exist
    results = {}
    
    for split in splits:
        candidates_path = Path(data_dir) / f'candidates_{split}.parquet'
        
        if not candidates_path.exists():
            logger.warning(f"Candidates file not found for {split} split: {candidates_path}")
            continue
        
        logger.info(f"Evaluating reranking on {split} split")
        
        # Load candidates
        candidates_df = pd.read_parquet(candidates_path)
        
        if sample_size and len(candidates_df) > sample_size:
            candidates_df = candidates_df.sample(n=sample_size, random_state=42)
        
        # Simple reranking evaluation (without loading the actual model)
        # This would need the actual model loading implementation
        logger.info(f"Loaded {len(candidates_df)} candidate queries for {split}")
        
        # Placeholder results - in practice, you'd load the model and run inference
        split_results = {}
        for k in k_values:
            # Mock results - replace with actual reranking evaluation
            split_results[f'recall@{k}'] = 0.0
            split_results[f'ndcg@{k}'] = 0.0
        
        results[split] = split_results
    
    return results


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level, rich_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evaluation pipeline")
    
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Initialize paths
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Determine models to evaluate
        models_to_evaluate = args.models
        if args.baseline_only:
            models_to_evaluate = ["baselines"]
        
        # Initialize results
        all_results = {
            'evaluation_config': {
                'data_dir': str(data_dir),
                'model_dir': args.model_dir,
                'k_values': args.k_values,
                'splits': args.splits,
                'models': models_to_evaluate,
                'sample_size': args.sample_size,
                'seed': args.seed,
            },
            'results': {}
        }
        
        # Evaluate baseline models
        if "baselines" in models_to_evaluate:
            logger.info("="*50)
            logger.info("EVALUATING BASELINE MODELS")
            logger.info("="*50)
            
            baseline_results = evaluate_baselines(
                data_dir, args.k_values, args.splits, args.sample_size
            )
            all_results['results']['baselines'] = baseline_results
            
            # Log baseline results
            for split, split_results in baseline_results.items():
                logger.info(f"\nBaseline Results - {split.upper()} Split:")
                for model_name, metrics in split_results.items():
                    logger.info(f"  {model_name}:")
                    for k in args.k_values:
                        if f'Recall' in metrics and k in metrics['Recall']:
                            recall = metrics['Recall'][k]
                            ndcg = metrics['NDCG'][k] if 'NDCG' in metrics and k in metrics['NDCG'] else 0.0
                            logger.info(f"    Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")
        
        # Evaluate retrieval model
        if "retrieval" in models_to_evaluate and args.model_dir:
            logger.info("="*50)
            logger.info("EVALUATING RETRIEVAL MODEL")
            logger.info("="*50)
            
            retrieval_results = evaluate_retrieval_model(
                data_dir, args.model_dir, args.k_values, args.splits, args.sample_size
            )
            all_results['results']['retrieval'] = retrieval_results
            
            # Log retrieval results
            for split, metrics in retrieval_results.items():
                logger.info(f"\nRetrieval Results - {split.upper()} Split:")
                for k in args.k_values:
                    if f'Recall' in metrics and k in metrics['Recall']:
                        recall = metrics['Recall'][k]
                        ndcg = metrics['NDCG'][k] if 'NDCG' in metrics and k in metrics['NDCG'] else 0.0
                        logger.info(f"  Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")
        
        # Evaluate reranking model
        if "reranking" in models_to_evaluate and args.model_dir:
            logger.info("="*50)
            logger.info("EVALUATING RERANKING MODEL")
            logger.info("="*50)
            
            reranking_results = evaluate_reranking_model(
                data_dir, args.model_dir, args.k_values, args.splits, args.sample_size
            )
            all_results['results']['reranking'] = reranking_results
        
        # Save results
        results_path = output_dir / 'evaluation_results.json'
        write_json(all_results, results_path)
        logger.info(f"Saved evaluation results to {results_path}")
        
        # Generate summary report
        summary_path = output_dir / 'evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("EVALUATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data Directory: {data_dir}\n")
            f.write(f"Model Directory: {args.model_dir}\n")
            f.write(f"K Values: {args.k_values}\n")
            f.write(f"Splits: {args.splits}\n")
            f.write(f"Sample Size: {args.sample_size}\n")
            f.write(f"Seed: {args.seed}\n\n")
            
            # Write detailed results
            for model_type, model_results in all_results['results'].items():
                f.write(f"{model_type.upper()} RESULTS\n")
                f.write("-" * 30 + "\n")
                
                for split, split_results in model_results.items():
                    f.write(f"\n{split.upper()} Split:\n")
                    
                    if model_type == "baselines":
                        for baseline_name, metrics in split_results.items():
                            f.write(f"  {baseline_name}:\n")
                            for k in args.k_values:
                                if 'Recall' in metrics and k in metrics['Recall']:
                                    recall = metrics['Recall'][k]
                                    ndcg = metrics['NDCG'][k] if 'NDCG' in metrics and k in metrics['NDCG'] else 0.0
                                    f.write(f"    Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}\n")
                    else:
                        for k in args.k_values:
                            if f'Recall' in split_results and k in split_results['Recall']:
                                recall = split_results['Recall'][k]
                                ndcg = split_results['NDCG'][k] if 'NDCG' in split_results and k in split_results['NDCG'] else 0.0
                                f.write(f"  Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}\n")
                
                f.write("\n")
        
        logger.info(f"Saved evaluation summary to {summary_path}")
        
        logger.info("Evaluation completed successfully!")
        
        # Print final summary
        print("\n" + "="*50)
        print("EVALUATION COMPLETED")
        print("="*50)
        print(f"Models evaluated: {', '.join(models_to_evaluate)}")
        print(f"Splits evaluated: {', '.join(args.splits)}")
        print(f"K values: {args.k_values}")
        print(f"Results saved to: {output_dir}")
        print("="*50)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

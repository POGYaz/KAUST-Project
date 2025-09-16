#!/usr/bin/env python3
"""
Build ANN index from item embeddings.

This script loads item embeddings (produced by the Two-Tower retriever),
builds an ANN index (FAISS if available, otherwise exact search fallback),
and persists the index artifacts to disk.

Example usage:
    python scripts/build_index.py --embeddings models/retriever/item_embeddings.npy --out models/retriever/ann_index
    python scripts/build_index.py --config configs/retriever.yaml --out models/retriever/ann_index --index-type ivf --n-lists 1024 --n-probe 16
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.inference.ann_index import build_index_from_embeddings
from src.utils.config import load_config
from src.utils.io import read_numpy
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and persist ANN index from item embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to retriever config (optional)")
    parser.add_argument(
        "--embeddings", type=str,
        help="Path to item embeddings .npy (overrides config if provided)"
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output prefix for index artifacts (without extension)"
    )
    parser.add_argument(
        "--index-type", type=str, default="flat", choices=["flat", "ivf", "hnsw"],
        help="ANN index type"
    )
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"], help="Similarity metric")
    parser.add_argument("--n-lists", type=int, default=None, help="Number of IVF lists (ivf only)")
    parser.add_argument("--n-probe", type=int, default=None, help="Number of IVF probes at search time (ivf only)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Resolve embeddings path
    embeddings_path: Path
    if args.embeddings:
        embeddings_path = Path(args.embeddings)
    elif args.config:
        cfg = load_config(args.config)
        # Prefer explicit model output dir from config
        out_dir = Path(cfg.get("output_dir", "models/retriever"))
        embeddings_path = out_dir / cfg.get("embeddings", {}).get("item_embeddings_file", "item_embeddings.npy")
    else:
        # Sensible default
        embeddings_path = Path("models/retriever/item_embeddings.npy")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Item embeddings not found: {embeddings_path}")

    logger.info(f"Loading item embeddings from {embeddings_path}")
    item_embeddings: np.ndarray = read_numpy(embeddings_path)
    logger.info(f"Item embeddings shape: {item_embeddings.shape}")

    logger.info(f"Building index type={args.index_type}, metric={args.metric}")
    index = build_index_from_embeddings(
        embeddings=item_embeddings,
        index_type=args.index_type,
        metric=args.metric,
        n_lists=args.n_lists,
        n_probe=args.n_probe,
    )

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    index.save(out_prefix)

    stats = index.get_stats()
    logger.info(f"Index built and saved at {out_prefix}")
    logger.info(f"Index stats: {stats}")

    # Print reproducible command
    cmd = (
        f"python scripts/build_index.py --embeddings {embeddings_path} --out {out_prefix} "
        f"--index-type {args.index_type} --metric {args.metric}"
    )
    if args.n_lists:
        cmd += f" --n-lists {args.n_lists}"
    if args.n_probe:
        cmd += f" --n-probe {args.n_probe}"
    logger.info(f"Reproduce with:\n{cmd}")


if __name__ == "__main__":
    main()

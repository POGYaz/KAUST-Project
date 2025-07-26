#!/usr/bin/env python3
"""
Script to create interaction matrices from engineered datasets.

This script loads the engineered features and creates various types of interaction matrices
suitable for recommendation systems (particularly ALS-based models).

Usage:
    python scripts/create_interaction_matrices.py
"""

import sys
from pathlib import Path

# Add src to path for importing modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "src"))

from models.interaction_matrix import InteractionMatrixBuilder


def main():
    """Main function to create interaction matrices."""
    print("="*60)
    print("JARIR RECOMMENDATION SYSTEM")
    print("Interaction Matrix Creation")
    print("="*60)
    
    try:
        # Initialize the builder
        print("\n1. Initializing InteractionMatrixBuilder...")
        builder = InteractionMatrixBuilder()
        
        # Load the engineered features
        print("\n2. Loading engineered features...")
        features_df = builder.load_data()
        
        print(f"\nDataset Overview:")
        print(f"  - Total interactions: {len(features_df):,}")
        print(f"  - Unique customers: {features_df['CustomerID'].nunique():,}")
        print(f"  - Unique products (SKUs): {features_df['SKU'].nunique():,}")
        print(f"  - Date range: {features_df.get('Date', 'N/A')}")
        
        # Show available features
        print(f"\nAvailable features:")
        for col in features_df.columns:
            print(f"  - {col}")
        
        # Create all interaction matrices
        print("\n3. Creating interaction matrices...")
        matrices = builder.create_all_matrices()
        
        # Print detailed statistics
        print("\n4. Matrix Statistics:")
        builder.print_matrix_stats()
        
        # Save matrices to disk
        print("\n5. Saving matrices...")
        builder.save_matrices()
        
        # Show how to access specific matrices
        print("\n6. Matrix Access Examples:")
        print("\nYou can now access matrices like this:")
        print("  - quantity_matrix = builder.interaction_matrices['quantity']")
        print("  - confidence_matrix = builder.interaction_matrices['confidence']")
        print("  - binary_matrix = builder.interaction_matrices['binary']")
        
        # Get user-item mappings
        mappings = builder.get_user_item_mapping()
        print(f"\n7. ID Mapping Examples:")
        print(f"  - First 3 customers: {list(mappings['user_to_customer'].items())[:3]}")
        print(f"  - First 3 SKUs: {list(mappings['item_to_sku'].items())[:3]}")
        
        print("\n" + "="*60)
        print("SUCCESS! Interaction matrices created and saved.")
        print("="*60)
        print(f"\nMatrices saved to: {project_root}/data/matrices/")
        print("You can now use these matrices for training recommendation models.")
        
        # Example of how to load matrices later
        print(f"\nTo load matrices later:")
        print(f"  from models.interaction_matrix import InteractionMatrixBuilder")
        print(f"  builder = InteractionMatrixBuilder()")
        print(f"  matrix = builder.load_matrix('confidence')")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have run the feature engineering step first.")
        print("The engineered features file should be at data/features/features_table_v2.csv")
        return 1
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
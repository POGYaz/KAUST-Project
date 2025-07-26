import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class InteractionMatrixBuilder:
    """
    Build various interaction matrices from engineered feature datasets.
    Supports multiple interaction types and sparse matrix formats.
    """
    
    def __init__(self, features_path=None):
        """
        Initialize the interaction matrix builder.
        
        Args:
            features_path: Path to the engineered features CSV file
        """
        self.features_path = features_path or self._get_default_features_path()
        self.features_df = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.interaction_matrices = {}
        
    def _get_default_features_path(self):
        """Get default path to the latest engineered features."""
        root = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
        features_dir = root / "data" / "features"
        
        # Use the latest version of features
        if (features_dir / "features_table_v2.csv").exists():
            return features_dir / "features_table_v2.csv"
        elif (features_dir / "features_table.csv").exists():
            return features_dir / "features_table.csv"
        else:
            raise FileNotFoundError("No engineered features found. Run feature engineering first.")
    
    def load_data(self):
        """Load the engineered features dataset."""
        logger.info(f"Loading features from {self.features_path}")
        
        self.features_df = pd.read_csv(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} customer-item interactions")
        logger.info(f"Unique customers: {self.features_df['CustomerID'].nunique()}")
        logger.info(f"Unique items (SKUs): {self.features_df['SKU'].nunique()}")
        
        # Encode users and items to sequential integers
        self.features_df['user_id'] = self.user_encoder.fit_transform(self.features_df['CustomerID'])
        self.features_df['item_id'] = self.item_encoder.fit_transform(self.features_df['SKU'])
        
        return self.features_df
    
    def create_quantity_matrix(self):
        """Create interaction matrix based on raw quantity."""
        logger.info("Creating quantity-based interaction matrix")
        
        users = self.features_df['user_id'].values
        items = self.features_df['item_id'].values
        quantities = self.features_df['quantity'].values
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        matrix = csr_matrix((quantities, (users, items)), shape=(n_users, n_items))
        self.interaction_matrices['quantity'] = matrix
        
        logger.info(f"Quantity matrix shape: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        return matrix
    
    def create_confidence_matrix(self):
        """Create interaction matrix based on confidence weights (c_ui)."""
        logger.info("Creating confidence-weighted interaction matrix")
        
        users = self.features_df['user_id'].values
        items = self.features_df['item_id'].values
        
        # Use c_ui if available, otherwise calculate it
        if 'c_ui' in self.features_df.columns:
            confidence = self.features_df['c_ui'].values
        else:
            logger.warning("c_ui not found, calculating confidence as 1 + 40 * quantity")
            confidence = 1 + 40 * self.features_df['quantity'].values
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        matrix = csr_matrix((confidence, (users, items)), shape=(n_users, n_items))
        self.interaction_matrices['confidence'] = matrix
        
        logger.info(f"Confidence matrix shape: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        return matrix
    
    def create_time_decayed_matrix(self):
        """Create interaction matrix with time decay applied."""
        logger.info("Creating time-decayed interaction matrix")
        
        users = self.features_df['user_id'].values
        items = self.features_df['item_id'].values
        
        # Use decayed_quantity if available, otherwise calculate it
        if 'decayed_quantity' in self.features_df.columns:
            decayed_values = self.features_df['decayed_quantity'].values
        else:
            logger.warning("decayed_quantity not found, calculating as quantity * exp(-0.01 * recency)")
            decay_rate = 0.01
            decayed_values = (self.features_df['quantity'] * 
                            np.exp(-decay_rate * self.features_df['recency'])).values
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        matrix = csr_matrix((decayed_values, (users, items)), shape=(n_users, n_items))
        self.interaction_matrices['time_decayed'] = matrix
        
        logger.info(f"Time-decayed matrix shape: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        return matrix
    
    def create_binary_matrix(self):
        """Create binary interaction matrix (1 if interaction exists, 0 otherwise)."""
        logger.info("Creating binary interaction matrix")
        
        users = self.features_df['user_id'].values
        items = self.features_df['item_id'].values
        ones = np.ones(len(users))
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        matrix = csr_matrix((ones, (users, items)), shape=(n_users, n_items))
        self.interaction_matrices['binary'] = matrix
        
        logger.info(f"Binary matrix shape: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        return matrix
    
    def create_normalized_matrix(self, matrix_type='quantity'):
        """Create normalized interaction matrix (row-wise normalization)."""
        logger.info(f"Creating normalized {matrix_type} interaction matrix")
        
        if matrix_type not in self.interaction_matrices:
            logger.error(f"Matrix type '{matrix_type}' not found. Create it first.")
            return None
        
        matrix = self.interaction_matrices[matrix_type].copy()
        
        # Row-wise normalization
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        row_diag = csr_matrix((1.0 / row_sums, (range(len(row_sums)), range(len(row_sums)))))
        normalized_matrix = row_diag @ matrix
        
        self.interaction_matrices[f'{matrix_type}_normalized'] = normalized_matrix
        
        logger.info(f"Normalized {matrix_type} matrix created")
        return normalized_matrix
    
    def create_all_matrices(self):
        """Create all types of interaction matrices."""
        logger.info("Creating all interaction matrix types")
        
        if self.features_df is None:
            self.load_data()
        
        # Create base matrices
        self.create_quantity_matrix()
        self.create_confidence_matrix()
        self.create_time_decayed_matrix()
        self.create_binary_matrix()
        
        # Create normalized versions
        self.create_normalized_matrix('quantity')
        self.create_normalized_matrix('confidence')
        
        logger.info(f"Created {len(self.interaction_matrices)} interaction matrices")
        return self.interaction_matrices
    
    def save_matrices(self, output_dir=None):
        """Save interaction matrices to disk."""
        if output_dir is None:
            root = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
            output_dir = root / "data" / "matrices"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving interaction matrices to {output_dir}")
        
        # Save matrices in NPZ format
        for matrix_name, matrix in self.interaction_matrices.items():
            matrix_path = output_dir / f"{matrix_name}_matrix.npz"
            np.savez_compressed(matrix_path, 
                              data=matrix.data,
                              indices=matrix.indices,
                              indptr=matrix.indptr,
                              shape=matrix.shape)
            logger.info(f"Saved {matrix_name} matrix to {matrix_path}")
        
        # Save encoders for mapping back to original IDs
        encoders_path = output_dir / "encoders.npz"
        np.savez(encoders_path,
                user_classes=self.user_encoder.classes_,
                item_classes=self.item_encoder.classes_)
        logger.info(f"Saved encoders to {encoders_path}")
        
        # Save metadata
        metadata = {
            'n_users': len(self.user_encoder.classes_),
            'n_items': len(self.item_encoder.classes_),
            'n_interactions': len(self.features_df),
            'matrix_types': list(self.interaction_matrices.keys())
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_path = output_dir / "matrices_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_matrix(self, matrix_name, matrices_dir=None):
        """Load a specific interaction matrix from disk."""
        if matrices_dir is None:
            root = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
            matrices_dir = root / "data" / "matrices"
        
        matrices_dir = Path(matrices_dir)
        matrix_path = matrices_dir / f"{matrix_name}_matrix.npz"
        
        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        logger.info(f"Loading {matrix_name} matrix from {matrix_path}")
        
        npz_file = np.load(matrix_path)
        matrix = csr_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']), 
                           shape=npz_file['shape'])
        
        return matrix
    
    def get_user_item_mapping(self):
        """Get mapping between encoded IDs and original CustomerID/SKU."""
        return {
            'user_to_customer': dict(zip(range(len(self.user_encoder.classes_)), 
                                       self.user_encoder.classes_)),
            'customer_to_user': dict(zip(self.user_encoder.classes_, 
                                       range(len(self.user_encoder.classes_)))),
            'item_to_sku': dict(zip(range(len(self.item_encoder.classes_)), 
                                  self.item_encoder.classes_)),
            'sku_to_item': dict(zip(self.item_encoder.classes_, 
                                  range(len(self.item_encoder.classes_))))
        }
    
    def print_matrix_stats(self):
        """Print statistics for all created matrices."""
        if not self.interaction_matrices:
            logger.warning("No matrices created yet")
            return
        
        print("\n" + "="*60)
        print("INTERACTION MATRIX STATISTICS")
        print("="*60)
        
        for name, matrix in self.interaction_matrices.items():
            density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
            print(f"\n{name.upper()} MATRIX:")
            print(f"  Shape: {matrix.shape}")
            print(f"  Non-zero entries: {matrix.nnz:,}")
            print(f"  Density: {density:.6f}")
            print(f"  Min value: {matrix.data.min():.3f}")
            print(f"  Max value: {matrix.data.max():.3f}")
            print(f"  Mean value: {matrix.data.mean():.3f}")


def main():
    """Main function to create interaction matrices."""
    # Initialize the builder
    builder = InteractionMatrixBuilder()
    
    # Load data and create all matrices
    builder.load_data()
    builder.create_all_matrices()
    
    # Print statistics
    builder.print_matrix_stats()
    
    # Save matrices
    builder.save_matrices()
    
    # Example: Get mappings
    mappings = builder.get_user_item_mapping()
    logger.info(f"Example customer mapping: {list(mappings['user_to_customer'].items())[:3]}")
    logger.info(f"Example SKU mapping: {list(mappings['item_to_sku'].items())[:3]}")


if __name__ == "__main__":
    main() 
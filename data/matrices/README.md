# Interaction Matrices

This directory contains interaction matrices created from the engineered Jarir recommendation system datasets.

## Files Overview

### Matrix Files (NPZ format)
- `quantity_matrix.npz` - Raw purchase quantities per customer-item pair
- `confidence_matrix.npz` - Confidence-weighted matrix (1 + α × quantity, α=40)
- `time_decayed_matrix.npz` - Time-decayed quantities (recent purchases weighted higher)
- `binary_matrix.npz` - Binary interaction matrix (1 if purchased, 0 otherwise)
- `quantity_normalized_matrix.npz` - Row-normalized quantity matrix
- `confidence_normalized_matrix.npz` - Row-normalized confidence matrix

### Supporting Files
- `encoders.npz` - Label encoders for mapping between original IDs and matrix indices
- `matrices_metadata.csv` - Summary metadata about all matrices

## Matrix Specifications

**Dimensions**: 800 users × 370 items  
**Total Interactions**: 43,720  
**Density**: ~14.77% (sparse matrices)  
**Format**: Compressed Sparse Row (CSR) matrices

## Usage

### Loading Matrices

```python
from src.models.interaction_matrix import InteractionMatrixBuilder

# Initialize builder
builder = InteractionMatrixBuilder()

# Load specific matrix
confidence_matrix = builder.load_matrix('confidence')
quantity_matrix = builder.load_matrix('quantity')
```

### ID Mappings

```python
import numpy as np

# Load encoders
encoders = np.load('data/matrices/encoders.npz')
user_classes = encoders['user_classes']  # Original CustomerIDs
item_classes = encoders['item_classes']  # Original SKUs

# Map matrix index to original ID
customer_id = user_classes[user_index]
sku = item_classes[item_index]
```

### Matrix Types Explained

1. **Quantity Matrix**: 
   - Raw purchase quantities
   - Values: 1-168 units
   - Best for: Understanding actual purchase volumes

2. **Confidence Matrix**: 
   - Calculated as: 1 + 40 × quantity
   - Values: 41-6,721
   - Best for: ALS recommendation algorithms (implicit feedback)

3. **Time-Decayed Matrix**: 
   - Calculated as: quantity × exp(-0.01 × recency_days)
   - Values: 0-134
   - Best for: Emphasizing recent purchase behavior

4. **Binary Matrix**: 
   - Simple binary interactions (1 if purchased)
   - Values: 0 or 1
   - Best for: Algorithms that work with binary feedback

5. **Normalized Matrices**: 
   - Row-normalized versions of quantity/confidence
   - Values: 0-1 (probabilities)
   - Best for: Algorithms requiring normalized inputs

## Recommendation Use Cases

### For ALS (Alternating Least Squares)
Use `confidence_matrix.npz` - designed specifically for implicit feedback ALS algorithms.

### For Matrix Factorization
Use `quantity_matrix.npz` or `confidence_matrix.npz` depending on whether you want raw or weighted values.

### For Binary Classification
Use `binary_matrix.npz` for algorithms that work with binary feedback.

### For Time-Sensitive Recommendations
Use `time_decayed_matrix.npz` to emphasize recent purchases over older ones.

## Dataset Statistics

- **Users**: 800 business customers (B2B-2000 through B2B-2799)
- **Items**: 370 unique SKUs (products)
- **Categories**: Computer Supplies, Office Supplies, School Supplies, Smart TV
- **Interaction Range**: 1-168 units per transaction
- **Time Period**: Covered in the original transaction data

## Generated From

These matrices were created from engineered features in:
- `data/features/features_table_v2.csv`

Using the script:
- `scripts/create_interaction_matrices.py`

With the core logic in:
- `src/models/interaction_matrix.py`

## Example Analysis

For detailed usage examples, see:
- `notebooks/Interaction_Matrix_Demo.ipynb`

This notebook shows how to:
- Load and analyze matrices
- Understand user behavior patterns  
- Create simple recommendations
- Compare different matrix types
- Visualize interaction patterns

## Technical Notes

- All matrices are stored in compressed NPZ format for efficiency
- Memory usage: ~1.5MB total for all matrices
- Matrices use 0-based indexing for users and items
- Sparse format (CSR) used to handle the ~85% zero values efficiently

## Next Steps

These matrices are ready for use with recommendation algorithms like:
- Implicit ALS (`implicit` library)
- Surprise library algorithms
- Custom collaborative filtering implementations
- Deep learning recommendation models (after conversion to appropriate format) 
# Jarir B2B Recommendation System

A production-ready recommendation system built with a modern two-stage machine learning pipeline. This system provides personalized product recommendations for B2B customers using advanced neural architectures and feature engineering.

## 🎯 Overview

The recommendation system employs a sophisticated two-stage approach:

1. **Retriever Stage**: Two-Tower neural network with user and item embeddings to efficiently identify top candidate products
2. **Reranker Stage**: Multi-layer perceptron (MLP) with engineered features to precisely rank the final recommendations

This architecture balances computational efficiency with recommendation quality, making it suitable for production deployment.

## 🚀 Live Demo

**Option 1: Streamlit Cloud (Recommended)**
- Visit the live application: [Jarir B2B Recommendation System](https://jarir-project.streamlit.app/)
- No installation required - ready to use immediately

**Option 2: Local Development**
- Clone and run locally for development and customization

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │   Two-Tower      │ -> │   MLP Reranker  │
│                 │    │   Retriever      │    │                 │
│ Customer ID +   │    │                  │    │ Features:       │
│ Preferences     │    │ Retrieves ~200   │    │ • Similarity    │
│                 │    │ candidates       │    │ • Popularity    │
│                 │    │                  │    │ • Price         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        v
                                               ┌─────────────────┐
                                               │ Top-K Products  │
                                               │ Ranked by       │
                                               │ Relevance       │
                                               └─────────────────┘
```

## 📁 Project Structure

```
KAUST-Project/
├── app/
│   └── streamlit_app.py          # Interactive web application
├── models/
│   ├── data/                     # Clean dataset (Parquet files)
│   │   ├── customers_clean.parquet
│   │   ├── items_clean.parquet
│   │   ├── interactions_clean.parquet
│   │   └── *_id_map.parquet
│   ├── retriever/                # Two-Tower model artifacts
│   │   ├── user_embeddings.npy
│   │   ├── item_embeddings.npy
│   │   └── training_metrics.json
│   └── reranker/                 # MLP reranker artifacts
│       ├── ranker_best.pt
│       └── training_metrics.json
├── src/                          # Core ML modules
│   ├── data/                     # Data processing & feature engineering
│   ├── models/                   # Model architectures
│   ├── training/                 # Training pipelines
│   ├── inference/                # Inference utilities
│   └── evaluation/               # Metrics & evaluation
├── configs/                      # YAML configuration files
├── scripts/                      # Training & data preparation CLIs
├── tests/                        # Unit tests
└── requirements.txt              # Dependencies
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/POGYaz/KAUST-Project
cd KAUST-Project
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model artifacts**
Ensure these files exist:
- `models/retriever/{user_embeddings.npy, item_embeddings.npy}`
- `models/reranker/ranker_best.pt`
- `models/data/*.parquet`

5. **Launch the application**
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 📊 Features

### 🎨 User Interface
- **Clean, professional design** with light/dark theme support
- **Customer selection** with smart search and profile insights
- **Real-time recommendations** with sub-second response times
- **Detailed explanations** for recommendation logic
- **Purchase history analysis** and behavioral insights

### 🤖 Machine Learning
- **Two-Tower retriever** with 128-dimensional embeddings
- **Feature-rich reranker** with engineered signals
- **FAISS-powered** approximate nearest neighbor search
- **Comprehensive evaluation** with Recall@K, NDCG@K metrics

### 🔧 Technical Features
- **Models-first architecture** for easy deployment
- **Automatic model compatibility** handling
- **Robust error handling** and fallback mechanisms
- **Performance monitoring** and debug information

## 📈 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Recall@10 | ~0.69 | Fraction of relevant items in top-10 |
| NDCG@10 | ~0.55 | Normalized discounted cumulative gain |
| Response Time | <1s | End-to-end recommendation latency |
| Model Size | ~50MB | Combined model artifacts |

## 🔄 Training Pipeline (Optional)

For retraining or customization:

1. **Data preparation**
```bash
python scripts/prepare_data.py --config configs/data.yaml
```

2. **Train retriever**
```bash
python scripts/train_retriever.py --config configs/retriever.yaml
```

3. **Train reranker**
```bash
python scripts/train_reranker.py --config configs/reranker.yaml
```

All outputs are automatically saved to the `models/` directory.

## 🚀 Deployment

### Streamlit Community Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy directly - no additional configuration needed

### Custom Infrastructure
- **Docker**: Use provided containerization (optional)
- **Cloud platforms**: AWS, GCP, Azure compatible
- **Requirements**: Python 3.10+, 2GB+ RAM, 1 CPU core

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

## 🛠️ Tech Stack

- **Core ML**: Python, PyTorch, NumPy, Pandas, scikit-learn
- **Data Processing**: PyArrow (Parquet), YAML configuration
- **Models**: Two-Tower neural retriever, MLP reranker with feature engineering
- **Search**: FAISS for approximate nearest neighbors (ANN)
- **UI/UX**: Streamlit with custom CSS, responsive design, light/dark themes
- **Development**: Rich logging, pytest testing, modular architecture
- **Deployment**: Models-first structure, Streamlit Community Cloud ready

## 🔍 Troubleshooting

### Common Issues

**Missing model files**
- Verify all required files exist in `models/` directory
- Check file permissions and paths

**Model loading errors**
- Clear Streamlit cache: Settings → Clear cache
- Restart the application

**Performance issues**
- Ensure sufficient RAM (8GB+ recommended)
- Check CPU usage during inference

**Import errors on Streamlit Cloud**
- The app automatically handles Python path configuration
- Verify all dependencies are in `requirements.txt`

## 🤝 Contributing

**Yazan Alkamal** | Software Engineering at UQU
- Led design and development of two-tower recommendation architecture
- Architected production-ready Streamlit application with business-focused interface
- Implemented models-first deployment structure and compatibility solutions
- Contributed to ML pipeline optimization and comprehensive data analysis features
- Enhanced project architecture and streamlined development workflow

# Feature Engineering Report

---

## Introduction

This report summarizes the set of features we engineered from our cleaned Transactions, Customers, and Products tables. These features are designed to provide a rich, business-relevant dataset for training an ALS (Alternating Least Squares) recommender system.

---

## Feature Descriptions

### 2.1 Raw Quantity

- **What it is:** Total units of each product (SKU) bought by each customer.
- **Why it matters:** Goes beyond “bought/not bought”—captures the strength of each customer–product relationship.

### 2.2 Recency

- **What it is:** Number of days since a customer’s most recent purchase.
- **Why it matters:** Recent activity better reflects current preferences; older orders are less informative.

### 2.3 Frequency

- **What it is:** Total number of orders ever placed by a customer.
- **Why it matters:** Highly active customers often follow more consistent patterns.

### 2.4 Monetary

- **What it is:** Total spend per customer (sum of unit price × quantity).
- **Why it matters:** High‑value accounts may have distinct buying behaviors.

### 2.5 Category Counts (`class_*`)

- **What it is:** For each revenue category, the count of that customer’s orders in that category.
- **Why it matters:** Directly indicates category preferences for each account.

### 2.6 SKU Summary

- **Interaction Count:** Number of customers who ordered each SKU.
- **Average Quantity:** Mean units per order of each SKU.
- **Average Price:** Mean unit price of each SKU.
- **Why it matters:** Provides item‑side popularity and size/price context, useful for weighting or cold‑start strategies.

### 2.7 Confidence Weight (`c_ui`)

- **What it is:** `1 + α × quantity` (with α = 40).
- **Why it matters:** Explicitly gives more weight to large purchases in an implicit‑feedback ALS model.

### 2.8 Time-Decayed Quantity

- **What it is:** `quantity × exp(–decay_rate × recency)` (decay\_rate = 0.01).
- **Why it matters:** Gradually reduces the influence of older orders, emphasizing recent activity.

### 2.9 Customer Tenure

- **What it is:** Days between a customer’s signup date and the most recent transaction date.
- **Why it matters:** Newer vs. long-standing accounts can behave differently.

### 2.10 Log-Transformed Metrics

- **What it is:** `log1p` transformation applied to monetary, interaction\_count, and avg\_price.
- **Why it matters:** Reduces the impact of extreme values and stabilizes variance for modeling.

---

## Data Quality

- **Completeness:** All key features (`quantity`, `recency`, `frequency`, `monetary`, category counts, SKU summaries) contain 0% missing values.
- **Consistency:** Datatypes and ranges have been validated. Numeric features have no invalid or negative entries, and categorical fields match expected enums.

---

## Recommendations & Next Steps

1. **Hyperparameter Tuning**
   - Experiment with different values of α (confidence weight) and decay\_rate to optimize model accuracy.
2. **Time-Window Features**
   - Add separate recency/frequency measurements over 30- and 90-day windows to capture both short- and long-term behavior.
3. **Feature Scaling**
   - Apply standardization or min-max scaling to continuous features if needed by your ALS implementation.
4. **Hybrid Signals**
   - Prepare side-features (e.g., text embeddings of product descriptions, firmographic enrichments) for potential extension to a hybrid recommender.
5. **Monitoring & Drift Detection**
   - Set up automated checks on key distributions (recency, quantity, price) to alert if data patterns shift over time.


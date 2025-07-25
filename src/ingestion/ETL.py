import numpy as np
import pandas as pd
import os
import logging
import datetime
import zoneinfo

# Set timezone for log timestamp
_timezone = zoneinfo.ZoneInfo('Asia/Riyadh')
_log_timestamp = datetime.datetime.now(tz=_timezone).strftime('%Y%m%d_%H%M%S')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(f"etl_pipeline_{_log_timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ETL:
    def __init__(self, customers_path, products_catalog_path, transactions_log_path, output_path):
        self.customers_path = customers_path
        self.products_catalog_path = products_catalog_path
        self.transactions_log_path = transactions_log_path
        self.output_path = output_path

    def load_and_process(self, threshold=2):
        """Handles full ETL: logging, loading, type casting, merging, saving"""
        
        customers = pd.read_csv(self.customers_path)
        products = pd.read_csv(self.products_catalog_path)
        transactions = pd.read_csv(self.transactions_log_path)
        logger.info("Loaded dataframes from CSVs")

        logger.info(f"* Before Merging *\ncustomers: {customers.shape}\nproducts: {products.shape}\ntransactions: {transactions.shape}")

        if 'Date' in transactions.columns and 'Timestamp' not in transactions.columns:
            transactions.rename(columns={'Date': 'Timestamp'}, inplace=True)

        customers['CustomerID'] = customers['CustomerID'].astype(str)
        customers['Business_Category'] = customers['Business_Category'].astype(str)
        customers['Business_Size'] = customers['Business_Size'].astype(str)
        customers['Customer_Since'] = pd.to_datetime(customers['Customer_Since'])

        products['SKU'] = products['SKU'].astype(str)
        products['Rev_GL_Class'] = products['Rev_GL_Class'].astype(str)
        products['Sub_Category'] = products['Sub_Category'].astype(str)
        products['Item_Description'] = products['Item_Description'].astype(str)
        products['Brand'] = products['Brand'].astype(str)
        products['Unit_Price'] = pd.to_numeric(products['Unit_Price'], errors='coerce')
        products['Attributes'] = products['Attributes'].astype(str)

        transactions['TransactionID'] = transactions['TransactionID'].astype(str)
        transactions['CustomerID'] = transactions['CustomerID'].astype(str)
        transactions['Timestamp'] = pd.to_datetime(transactions['Timestamp'])
        transactions['SKU'] = transactions['SKU'].astype(str)
        transactions['Quantity'] = pd.to_numeric(transactions['Quantity'], downcast='integer', errors='coerce')

        logger.info("Lightweight type casting completed")

        valid_cust_mask = transactions['CustomerID'].isin(customers['CustomerID'])
        invalid_cust_count = (~valid_cust_mask).sum()
        invalid_cust_percent = invalid_cust_count / len(transactions) * 100
        if invalid_cust_count > 0:
            if invalid_cust_percent < threshold:
                logger.warning(f"Dropping {invalid_cust_count} rows with invalid CustomerIDs (< {threshold}%)")
                transactions = transactions[valid_cust_mask]
            else:
                logger.warning(f"Too many invalid CustomerID rows ({invalid_cust_percent:.2f}%) — no rows dropped")
        else:
            logger.info("All CustomerIDs are valid.")

        valid_sku_mask = transactions['SKU'].isin(products['SKU'])
        invalid_sku_count = (~valid_sku_mask).sum()
        invalid_sku_percent = invalid_sku_count / len(transactions) * 100
        if invalid_sku_count > 0:
            if invalid_sku_percent < threshold:
                logger.warning(f"Dropping {invalid_sku_count} rows with invalid SKUs (< {threshold}%)")
                transactions = transactions[valid_sku_mask]
            else:
                logger.warning(f"Too many invalid SKU rows ({invalid_sku_percent:.2f}%) — no rows dropped")
        else:
            logger.info("All SKUs are valid.")

        merged_df = pd.merge(transactions, customers, on='CustomerID', how='left')
        merged_df = pd.merge(merged_df, products, on='SKU', how='left')
        merged_df = merged_df.sort_values(by=['CustomerID', 'Timestamp', 'SKU'])

        logger.info(f"* After Merging *\nmerged shape: {merged_df.shape}")

        mem_MB = merged_df.memory_usage(deep=True).sum() / 1_048_576
        logger.info(f"Estimated in-memory size: {mem_MB:.2f} MB")

        if mem_MB < 1000:
            merged_df.to_csv(self.output_path + ".csv", index=False)
            logger.info("Saved dataset as CSV")
        else:
            merged_df.to_parquet(self.output_path + ".parquet", index=False)
            logger.info("Saved dataset as Parquet")

        for handler in logger.handlers:
            handler.flush()

        return merged_df


# ✅ Optional: Only run this if the script is called directly
if __name__ == "__main__":
    import sys
    logger.info("Running ETL script directly.")
    if len(sys.argv) < 5:
        logger.error("Usage: python etl_pipeline.py <customers.csv> <products.csv> <transactions.csv> <output_path>")
        sys.exit(1)

    customers = sys.argv[1]
    products = sys.argv[2]
    transactions = sys.argv[3]
    output_path = sys.argv[4]

    etl = ETL(customers, products, transactions, output_path)
    etl.load_and_process()

import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.entity.config_entity import DataIngestionConfig

logging.basicConfig(level=logging.INFO)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _read_data(self):
        return pd.read_csv(self.config.source_data)

    def _save_raw_data(self, df):
        raw_data_path = Path(self.config.root_dir) / "raw.csv"
        df.to_csv(raw_data_path, index=False)
        return raw_data_path

    def _split_data(self, df):
        return train_test_split(df, test_size=0.2, random_state=42)

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")

        os.makedirs(self.config.root_dir, exist_ok=True)

        df = self._read_data()
        self._save_raw_data(df)

        train_df, test_df = self._split_data(df)

        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logging.info("Data ingestion completed")

        return self.config.train_data_path, self.config.test_data_path

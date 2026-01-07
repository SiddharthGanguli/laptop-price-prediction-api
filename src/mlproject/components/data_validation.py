import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from mlproject.entity.config_entity import DataValidationConfig

logging.basicConfig(level=logging.INFO)


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _read_schema(self):
        with open(self.config.schema_file, "r") as f:
            schema = yaml.safe_load(f)
        return schema
    
    def initiate_data_validation(self):
        logging.info("Starting data validation")

        schema = self._read_schema()
        expected_columns = list(schema["columns"].keys())

        df = pd.read_csv(self.config.train_data_path)
        actual_columns = list(df.columns)

        if len(expected_columns) != len(actual_columns):
            logging.error("Column count mismatch")
            status = False

        elif expected_columns != actual_columns:
            logging.error("Column names mismatch")
            status = False
        else:
            logging.info("Data validation passed")
            status = True

        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.status_file, "w") as f:
            f.write(str(status))

        return status
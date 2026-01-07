import logging
from pathlib import Path

from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_ingestion import DataIngestion

logging.basicConfig(level=logging.INFO)

STAGE_NAME = "Data Ingestion Stage"

def main():
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    config_manager = ConfigurationManager(
        config_filepath=Path("config/config.yaml")
    )

    data_ingestion_config = config_manager.get_data_ingestion_config()

    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.initiate_data_ingestion()

    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == "__main__":
    main()

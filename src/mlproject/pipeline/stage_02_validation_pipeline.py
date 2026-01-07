import logging
from pathlib import Path

from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_validation import DataValidation

logging.basicConfig(level=logging.INFO)

STAGE_NAME = "Data Validation Stage"

def main():
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    config_manager = ConfigurationManager(
        config_filepath=Path("config/config.yaml")
    )

    data_validation_config = config_manager.get_data_validation_config()

    data_validation = DataValidation(config=data_validation_config)
    status = data_validation.initiate_data_validation()

    if not status:
        raise Exception("Data validation failed")

    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == "__main__":
    main
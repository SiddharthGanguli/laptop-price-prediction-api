import logging
from pathlib import Path

from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_preproceing import DataTransformation

logging.basicConfig(level=logging.INFO)

STAGE_NAME = "Data Transformation Stage"

def main():
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    config_manager = ConfigurationManager(
        config_filepath=Path("config/config.yaml")
    )

    transformation_config = config_manager.get_data_transformation_config()

    data_transformation = DataTransformation(config=transformation_config)
    data_transformation.initiate_data_transformation()

    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == "__main__":
    main
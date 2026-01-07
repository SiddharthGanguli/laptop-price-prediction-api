import logging
from pathlib import Path

from mlproject.config.configuration import ConfigurationManager
from mlproject.components.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)

STAGE_NAME = "Model Training Stage"


def main():
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    config_manager = ConfigurationManager(
        config_filepath=Path("config/config.yaml")
    )

    model_trainer_config = config_manager.get_model_trainer_config()

    model_trainer = ModelTrainer(config=model_trainer_config)
    model_trainer.initiate_model_training()

    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == "__main__":
    main()

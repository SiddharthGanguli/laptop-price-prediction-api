import logging
from pathlib import Path

from mlproject.config.configuration import ConfigurationManager
from mlproject.components.model_evaluation import ModelEvaluation

logging.basicConfig(level=logging.INFO)

STAGE_NAME = "Model Evaluation Stage"


def main():
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    config_manager = ConfigurationManager(
        config_filepath=Path("config/config.yaml")
    )

    evaluation_config = config_manager.get_model_evaluation_config()

    model_evaluation = ModelEvaluation(config=evaluation_config)
    status = model_evaluation.initiate_model_evaluation()

    if not status:
        raise Exception("Model did not meet evaluation criteria")

    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == "__main__":
    main
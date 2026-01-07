import os
from pathlib import Path
import yaml

from mlproject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig


)

class ConfigurationManager:
    def __init__(self, config_filepath: Path):
        self.config = self._read_yaml(config_filepath)

    def _read_yaml(self, filepath: Path):
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion = self.config["data_ingestion"]

        os.makedirs(ingestion["root_dir"], exist_ok=True)

        return DataIngestionConfig(
            root_dir=Path(ingestion["root_dir"]),
            source_data=Path(ingestion["source_data"]),
            train_data_path=Path(ingestion["train_data_path"]),
            test_data_path=Path(ingestion["test_data_path"])
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        validation = self.config["data_validation"]

        os.makedirs(validation["root_dir"], exist_ok=True)

        return DataValidationConfig(
            root_dir=Path(validation["root_dir"]),
            train_data_path=Path(validation["train_data_path"]),
            schema_file=Path(validation["schema_file"]),
            status_file=Path(validation["status_file"])
        )
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation = self.config["data_transformation"]
        return DataTransformationConfig(
            root_dir=Path(transformation["root_dir"]),
            train_data_path=Path(transformation["train_data_path"]),
            test_data_path=Path(transformation["test_data_path"]),
            transformed_train_path=Path(transformation["transformed_train_path"]),
            transformed_test_path=Path(transformation["transformed_test_path"]),
            preprocessor_path=Path(transformation["preprocessor_path"]),
            target_column=transformation["target_column"],  # <-- Add this line
    )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer = self.config["model_trainer"]

        os.makedirs(trainer["root_dir"], exist_ok=True)

        return ModelTrainerConfig(
            root_dir=Path(trainer["root_dir"]),
            train_data_path=Path(trainer["train_data_path"]),
            test_data_path=Path(trainer["test_data_path"]),
            model_path=Path(trainer["model_path"]),
            experiment_name=trainer["experiment_name"],
            model_params=trainer["model_params"]
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation = self.config["model_evaluation"]

        os.makedirs(evaluation["root_dir"], exist_ok=True)

        return ModelEvaluationConfig(
            root_dir=Path(evaluation["root_dir"]),
            model_path=Path(evaluation["model_path"]),
            test_data_path=Path(evaluation["test_data_path"]),
            metric_file=Path(evaluation["metric_file"]),
            min_r2_score=evaluation["min_r2_score"]
        )

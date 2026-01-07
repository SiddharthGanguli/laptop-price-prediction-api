from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class DataIngestionConfig :
    root_dir : Path
    source_data :Path
    train_data_path :Path
    test_data_path :Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    train_data_path: Path
    schema_file: Path
    status_file: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path
    target_column: str  # <-- ADD THIS

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    experiment_name: str
    model_params: Dict

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    metric_file: Path
    min_r2_score: float


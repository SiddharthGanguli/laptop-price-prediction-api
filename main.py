import logging
from pathlib import Path

from src.mlproject.pipeline.stage_01_ingestion_pipeline import main as data_ingestion_main
from src.mlproject.pipeline.stage_02_validation_pipeline import main as data_validation_main
from src.mlproject.pipeline.stage_03_preprocessing_pipeline import main as data_transformation_main
from src.mlproject.pipeline.stage_04_traning_pipline import main as model_trainer_main
from src.mlproject.pipeline.stage_05_model_evaluation import main as model_evaluation_main

logging.basicConfig(level=logging.INFO)

STAGES = [
    ("Data Ingestion", data_ingestion_main),
    ("Data Validation", data_validation_main),
    ("Data Transformation", data_transformation_main),
    ("Model Training", model_trainer_main),
    ("Model Evaluation", model_evaluation_main),
]


def main():
    for stage_name, stage_func in STAGES:
        try:
            logging.info(f"\n\n===== Starting {stage_name} =====")
            stage_func()  # 👈 call the stage
            logging.info(f"===== Completed {stage_name} =====\n")
        except Exception as e:
            logging.error(f"❌ {stage_name} failed")
            logging.error(e)
            raise e


if __name__ == "__main__":
    main()

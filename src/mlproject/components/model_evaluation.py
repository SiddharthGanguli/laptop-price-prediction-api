import os
import yaml
import numpy as np
import joblib
import mlflow

from sklearn.metrics import mean_squared_error, r2_score

from mlproject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_model(self):
        return joblib.load(self.config.model_path)

    def _load_test_data(self):
        return np.load(self.config.test_data_path)

    def _split_features_target(self, array):
        X = array[:, :-1]
        y = array[:, -1]
        return X, y

    def initiate_model_evaluation(self):
        os.makedirs(self.config.root_dir, exist_ok=True)

        model = self._load_model()
        test_arr = self._load_test_data()

        X_test, y_test = self._split_features_target(test_arr)

        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)


        # log metrics to MLflow (same run lineage)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        metrics = {
            "rmse": rmse,
            "r2_score": r2
        }

        with open(self.config.metric_file, "w") as f:
            yaml.safe_dump(metrics, f)

        # decision
        is_model_accepted = r2 >= self.config.min_r2_score

        return is_model_accepted

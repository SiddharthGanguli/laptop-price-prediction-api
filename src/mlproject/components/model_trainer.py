import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor

from mlproject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_data(self):
        train_arr = np.load(self.config.train_data_path)
        test_arr = np.load(self.config.test_data_path)
        return train_arr, test_arr

    def _split_features_target(self, array):
        X = array[:, :-1]
        y = array[:, -1]
        return X, y

    def initiate_model_training(self):
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_arr, test_arr = self._load_data()

        X_train, y_train = self._split_features_target(train_arr)
        X_test, y_test = self._split_features_target(test_arr)

        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run():

            model = RandomForestRegressor(**self.config.model_params)
            model.fit(X_train, y_train)

            # log hyperparameters
            mlflow.log_params(self.config.model_params)

            # save model locally
            joblib.dump(model, self.config.model_path)

            # log model to MLflow
            mlflow.sklearn.log_model(model, "model")

        return self.config.model_path

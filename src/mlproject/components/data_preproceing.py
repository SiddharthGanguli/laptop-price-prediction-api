import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _read_data(self):
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        return train_df, test_df

    def _get_columns(self, df):
        categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        return categorical_cols, numerical_cols

    def _build_preprocessor(self, categorical_cols, numerical_cols):
        cat_pipeline = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_pipeline, categorical_cols),
                ("num", num_pipeline, numerical_cols)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self):
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_df, test_df = self._read_data()

        target_column = self.config.target_column

        # Split features & target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].to_numpy().reshape(-1, 1)

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].to_numpy().reshape(-1, 1)

        # Identify column types
        categorical_cols, numerical_cols = self._get_columns(X_train)

        # Build preprocessing pipeline
        preprocessor = self._build_preprocessor(
            categorical_cols,
            numerical_cols
        )

        # 🔥 FIX: convert sparse → dense
        X_train_transformed = preprocessor.fit_transform(X_train).toarray()
        X_test_transformed = preprocessor.transform(X_test).toarray()

        # Concatenate features & target
        train_arr = np.c_[X_train_transformed, y_train]
        test_arr = np.c_[X_test_transformed, y_test]

        # Save transformed arrays
        np.save(self.config.transformed_train_path, train_arr)
        np.save(self.config.transformed_test_path, test_arr)

        # Save preprocessor
        joblib.dump(preprocessor, self.config.preprocessor_path)

        return (
            self.config.transformed_train_path,
            self.config.transformed_test_path,
            self.config.preprocessor_path
        )

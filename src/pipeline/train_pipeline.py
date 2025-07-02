import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException
from src.utils import save_object  # Must be implemented in src/utils.py

# Paths
DATA_PATH = os.path.join("artifacts", "train.csv")
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

def build_preprocessor(numerical_cols, categorical_cols):
    try:
        # Create transformers
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine into one preprocessor
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys) from e

def train_pipeline():
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)

        # Separate features and target
        X = df.drop(columns=["math_score"])
        y = df["math_score"]

        # Feature columns
        categorical_cols = [
            "gender", 
            "race_ethnicity", 
            "parental_level_of_education", 
            "lunch", 
            "test_preparation_course"
        ]
        numerical_cols = ["reading_score", "writing_score"]

        # Build and fit preprocessor
        preprocessor = build_preprocessor(numerical_cols, categorical_cols)
        X_processed = preprocessor.fit_transform(X)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_processed, y)

        # Save model and preprocessor
        save_object(PREPROCESSOR_PATH, preprocessor)
        save_object(MODEL_PATH, model)

        # Optional: Evaluate
        y_pred = model.predict(X_processed)
        print("Training R²:", r2_score(y, y_pred))
        print("Training RMSE:", mean_squared_error(y, y_pred, squared=False))

        print("✅ Model and preprocessor saved to /artifacts")

    except Exception as e:
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    train_pipeline()

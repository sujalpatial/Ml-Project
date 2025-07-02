"""
src/pipeline/predict_pipeline.py
--------------------------------
Utilities for turning a single row of raw user input into a model
prediction.  Requires two pickled files in the project’s `artifacts/`
folder:

    • preprocessor.pkl  – the fitted ColumnTransformer / Pipeline
    • model.pkl         – the trained regression / classifier model
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from typing import Any

from src.exception import CustomException
from src.utils import load_object

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")  # ← fixed typo


class PredictPipeline:
    """Wrapper that loads the artefacts once and re‑uses them for every call."""

    def __init__(self) -> None:
        try:
            self.model = load_object(MODEL_PATH)
            self.preprocessor = load_object(PREPROCESSOR_PATH)
        except Exception as e:  # propagate with context
            raise CustomException(e, sys) from e

    def predict(self, features: pd.DataFrame) -> Any:
        """
        Parameters
        ----------
        features : pd.DataFrame
            One or more rows with the raw columns expected by the preprocessor.

        Returns
        -------
        preds : np.ndarray | list
            Model predictions.
        """
        try:
            # Sanity check
            if not hasattr(self.preprocessor, "transform"):
                raise AttributeError(
                    f"Loaded preprocessor object of type {type(self.preprocessor)} "
                    "has no `.transform()` method. Was it saved correctly?"
                )

            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    """
    Collects the raw inputs from a form / API call and converts them
    into the single‑row DataFrame expected by PredictPipeline.predict().
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Return the data as a one‑row DataFrame ready for prediction."""
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys) from e

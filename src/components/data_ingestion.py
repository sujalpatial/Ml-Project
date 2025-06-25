import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

print("sys.path:", sys.path)
print("os.getcwd():", os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join('artifacts', "train.csv")
    test_path: str = os.path.join('artifacts', "test.csv")
    raw_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(r'C:\ML Project\src\notebook\stud.csv')
            os.makedirs(os.path.dirname(self.config.raw_path), exist_ok=True)
            df.to_csv(self.config.raw_path, index=False)
            logging.info("Saved raw data")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.config.train_path, index=False)
            test.to_csv(self.config.test_path, index=False)
            logging.info("Saved train and test sets")

            return self.config.train_path, self.config.test_path

        except Exception as e:
            logging.error("Error in data ingestion", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate()
    print(f"Train path: {train_path}, Test path: {test_path}")
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)
    modeltrainer=ModelTrainer()
    dummy_preprocessor_path=None
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,dummy_preprocessor_path))
    
print("Script started")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging  # this should point to your logger that writes to a file

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            df = pd.read_csv(r'C:\ML Project\src\notebook\stud.csv')  # raw string used
            logging.info('Dataset read into DataFrame')

            # Create the directory only once for clarity
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Exception occurred in data ingestion", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    print("Starting data ingestion")
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    print("Data ingestion finished")

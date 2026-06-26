# Used for creating file and folder paths
import os

# Used for handling system-level exceptions
import sys

# Used for reading CSV files
import pandas as pd

# Used for splitting data into train and test sets
from sklearn.model_selection import train_test_split

# Used for creating configuration classes
from dataclasses import dataclass

# Custom exception class
from src.exception import CustomException

# Custom logger
from src.logger import logging

from src.components.data_validation import DataValidation

from src.components.data_transformation import DataTransformation,DataTransformationConfig

from src.components.model_trainer import ModelTrainer


# Configuration class to store all file paths
@dataclass
class DataIngestionConfig:

    # Folder where all generated files will be stored
    ingestion_artifacts_dir: str = os.path.join(
        "artifacts",
        "data_ingestion"
    )

    # Path to save raw dataset
    raw_data_path: str = os.path.join(
        ingestion_artifacts_dir,
        "raw.csv"
    )

    # Path to save training dataset
    train_data_path: str = os.path.join(
        ingestion_artifacts_dir,
        "train.csv"
    )

    # Path to save testing dataset
    test_data_path: str = os.path.join(
        ingestion_artifacts_dir,
        "test.csv"
    )


class DataIngestion:
    """
    Responsible for:

    1. Reading raw dataset
    2. Saving raw dataset
    3. Creating train-test split
    4. Saving train and test datasets
    """

    def __init__(self):

        # Create object of configuration class
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path):
        """
        Reads raw dataset and creates train-test split

        Parameters
        ----------
        file_path : str
            Path of the raw dataset

        Returns
        -------
        tuple
            train_data_path, test_data_path
        """

        logging.info("Data Ingestion Started")

        try:

            # Read raw dataset
            df = pd.read_csv(file_path)

            logging.info(
                f"Dataset loaded successfully with shape {df.shape}"
            )

            validator = DataValidation()

            validator.validate_dataset(file_path)

            # Create artifacts directory if not present
            os.makedirs(
                self.ingestion_config.ingestion_artifacts_dir,
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw dataset saved successfully"
            )

            # Split dataset into train and test
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42,

                # Important for Alzheimer's dataset
                stratify=df["Diagnosis"]
            )

            logging.info(
                "Train-Test split completed"
            )

            # Save training dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save testing dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Train and Test datasets saved successfully"
            )

            logging.info(
                f"Train Shape : {train_set.shape}"
            )

            logging.info(
                f"Test Shape : {test_set.shape}"
            )

            logging.info(
                "Data Ingestion Completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logging.error(
                "Exception occurred during Data Ingestion"
            )

            raise CustomException(e, sys)


# Used only for testing this file independently
if __name__ == "__main__":

    # Create DataIngestion object
    data_ingestion = DataIngestion()

    # Run ingestion process
    train_path, test_path = (
        data_ingestion.initiate_data_ingestion(
            file_path="notebooks\\data\\alzheimers_disease_data.csv"
        )
    )

    data_transformation = DataTransformation()
    train_array, test_array, _,feature_names=data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)

    print("Train File :", train_path)
    print("Test File  :", test_path)

    modelTrainer=ModelTrainer()

    modelTrainer.initiate_model_trainer(train_array=train_array,test_array=test_array,feature_names=feature_names)
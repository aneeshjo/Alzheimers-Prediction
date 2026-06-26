# ============================================================
# Import Required Libraries
# ============================================================

import os
import sys

# Custom Exception
from src.exception import CustomException

# Logging
from src.logger import logging

# Project Components
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.config import load_config




# ============================================================
# Training Pipeline
# ============================================================

class TrainingPipeline:
    """
    This class orchestrates the complete machine learning
    training workflow.

    Workflow

    Data Ingestion
            ↓
    Data Validation
            ↓
    Data Transformation
            ↓
    Model Training
    """

    def __init__(self):

        logging.info(
            "Training Pipeline Initialized."
        )


    def run_pipeline(self, dataset_path):

        """
        Execute the complete training pipeline.

        Parameters
        ----------
        dataset_path : str

            Path of the raw dataset.
        """

        try:

            logging.info("=" * 80)
            logging.info("Starting Training Pipeline")
            logging.info("=" * 80)

            # =====================================================
            # Step 1 : Data Ingestion
            # =====================================================

            logging.info("Step 1 : Data Ingestion")

            ingestion = DataIngestion()

            train_path, test_path = ingestion.initiate_data_ingestion(
                file_path=dataset_path
            )

            logging.info("Data Ingestion Completed.")

            # =====================================================
            # Step 2 : Data Validation
            # =====================================================

            logging.info("Step 2 : Data Validation")

            validation = DataValidation()

            validation.validate_dataset(
                file_path=dataset_path
            )

            logging.info("Data Validation Completed.")

            # =====================================================
            # Step 3 : Data Transformation
            # =====================================================

            logging.info("Step 3 : Data Transformation")

            transformation = DataTransformation()

            (
                train_array,
                test_array,
                preprocessor_path,
                feature_names
            ) = transformation.initiate_data_transformation(

                train_path=train_path,

                test_path=test_path

            )

            logging.info(
                "Data Transformation Completed."
            )

            # =====================================================
            # Step 4 : Model Training
            # =====================================================

            logging.info("Step 4 : Model Training")

            trainer = ModelTrainer()

            trainer.initiate_model_trainer(

                train_array=train_array,

                test_array=test_array,

                feature_names=feature_names

            )

            logging.info(
                "Model Training Completed."
            )

            logging.info("=" * 80)

            logging.info(
                "Training Pipeline Completed Successfully."
            )

            logging.info("=" * 80)


        except Exception as e:

            logging.error(
                "Training Pipeline Failed."
            )

            raise CustomException(
                e,
                sys
            )


# ============================================================
# Run Pipeline
# ============================================================

if __name__ == "__main__":

    pipeline = TrainingPipeline()

    config = load_config()

    pipeline.run_pipeline(
    dataset_path=config["dataset_path"]

    )
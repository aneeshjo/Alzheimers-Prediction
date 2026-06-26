import os
import sys
from src.exception import CustomException
import dill
from src.logger import logging
from src.utils import save_object, evaluate_model
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline


from lightgbm import LGBMClassifier
from sklearn.metrics import(
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training.
    It defines the path to save the trained model object.
    """
    modelTrainer_artifacts_dir: str = os.path.join(
        "artifacts",
        "model_training"
    )
    trained_model_file_path: str = os.path.join(modelTrainer_artifacts_dir, 'model.pkl') 

class ModelTrainer:
        
    """
        Class responsible for training the machine learning model.
        It defines a method to train the model and save the trained model object to a file.
    """
    
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        feature_names
        ):
        """
        Train the final LightGBM model and save it.

        Parameters
        ----------
        train_array : ndarray

        test_array : ndarray

        Returns
        -------
        dict
            Evaluation metrics
        """

        try:

            logging.info("Starting Model Training.")

            # ==============================
            # Split train and test arrays
            # ==============================

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Train and Test arrays prepared.")

            # ==============================
            # Create final model
            # ==============================

            model = LGBMClassifier(

                random_state=42

            )

            logging.info("Training LightGBM model.")

            # ==============================
            # Train model
            # ==============================

            model.fit(
                X_train,
                y_train
            )

            logging.info("Model training completed.")

            # Get feature names from the transformed training dataframe

            feature_importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            feature_importance.sort_values(
                by="Importance",
                ascending=False,
                inplace=True
            )

            feature_importance.to_csv(
                "artifacts/feature_importance.csv",
                index=False
            )

            # ==============================
            # Evaluate trained model
            # ==============================

            metrics = evaluate_model(

                model=model,

                X_test=X_test,

                y_test=y_test

            )

            # ==============================
            # Save trained model
            # ==============================

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=model

            )

            logging.info("Model saved successfully.")

            logging.info(f"Model Path : {self.model_trainer_config.trained_model_file_path}")

            return metrics

        except Exception as e:

            logging.error("Error occurred during model training.")

            raise CustomException(e, sys)
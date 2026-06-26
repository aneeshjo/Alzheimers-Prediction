# Some utility functions for the project
import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import(
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score
)

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.
    
    Parameters:
    - file_path: str, the path where the object should be saved
    - obj: the Python object to be saved
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path using dill
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}.")
    
    except Exception as e:
        logging.error("Error occurred while saving the object.")
        raise CustomException(e, sys)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : Trained machine learning model

    X_test : Test feature data

    y_test : Test target values

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    try:

        logging.info("Starting model evaluation.")

        # ==============================
        # Predict target labels
        # ==============================

        y_pred = model.predict(X_test)

        # ==============================
        # Predict probabilities
        # Used for ROC-AUC calculation
        # ==============================

        if hasattr(model, "predict_proba"):

            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

        else:

            roc_auc = roc_auc_score(
                y_test,
                y_pred
            )

        # ==============================
        # Calculate evaluation metrics
        # ==============================

        metrics = {

            "Accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "ROC AUC": roc_auc

        }

        # ==============================
        # Log model performance
        # ==============================

        logging.info("Model Evaluation Results")

        for metric, value in metrics.items():

            logging.info(f"{metric}: {value:.4f}")

        logging.info("Model evaluation completed successfully.")

        return metrics

    except Exception as e:

        logging.error("Error occurred while evaluating model.")

        raise CustomException(e, sys)
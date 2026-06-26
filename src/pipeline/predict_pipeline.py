# ============================================================
# Import Required Libraries
# ============================================================

# Used for handling file paths
import os

# Used for handling system-specific exceptions
import sys

# Used for reading JSON files
# import json

# Used for creating DataFrames
import pandas as pd

# Configuration class
from dataclasses import dataclass

# Custom exception class
from src.exception import CustomException

# Logging configuration
from src.logger import logging

# Utility function to load saved objects (.pkl files)
from src.utils import load_object

# ============================================================
# Prediction Pipeline Configuration
# ============================================================

@dataclass
class PredictionPipelineConfig:
    """
    Configuration class for the prediction pipeline.

    This class stores the paths of all artifacts required
    during prediction.
    """

    # Path to the trained LightGBM model
        # Path to the trained LightGBM model
    modelTrainer_artifacts_dir: str = os.path.join(
        "artifacts",
        "model_training"
    )
    trained_model_file_path: str = os.path.join(modelTrainer_artifacts_dir, 'model.pkl')

    # Path to the preprocessing pipeline
    transformation_artifacts_dir: str = os.path.join(
        "artifacts",
        "data_transformation"
    )
    preprocessor_obj_file_path= os.path.join(transformation_artifacts_dir,"preprocesor.pkl")

    # # Path to the default values used for missing features
    # default_values_path = os.path.join(
    #     "artifacts",
    #     "default_values.json"
    # )

# ============================================================
# Prediction Pipeline
# ============================================================

class PredictPipeline:
    """
    PredictPipeline is responsible for

    1. Loading the trained model
    2. Loading the preprocessing pipeline
    3. Transforming the incoming data
    4. Predicting Alzheimer's diagnosis
    5. Returning prediction and probability
    """
    def __init__(self):
        logging.info("Initializing Prediction Pipeline.")

        self.config = PredictionPipelineConfig()

        # Load trained model
        self.model = load_object(
            self.config.trained_model_file_path
        )

        logging.info("Model loaded successfully.")

        # Load preprocessor
        self.preprocessor = load_object(
            self.config.preprocessor_obj_file_path
        )

        logging.info("Preprocessor loaded successfully.")
    
    def predict(self, features):
        """
        Predict Alzheimer's diagnosis.

        Parameters
        ----------
        features : pandas.DataFrame

            DataFrame containing all model features.

        Returns
        -------
        prediction : int

            0 = No Alzheimer's

            1 = Alzheimer's
            probability : float

            Probability of Alzheimer's

        risk_level : str

            Low Risk

            Moderate Risk

            High Risk
        """

        try:

            logging.info("Starting prediction process.")

            # ----------------------------------------
            # Apply preprocessing pipeline
            # ----------------------------------------

            transformed_features = self.preprocessor.transform(
                features
            )

            logging.info(
                "Input data transformed successfully."
            )

             # ----------------------------------------
            # Predict class
            # ----------------------------------------

            prediction = self.model.predict(
                transformed_features
            )[0]

            # ----------------------------------------
            # Predict probability
            # ----------------------------------------

            probability = self.model.predict_proba(
                transformed_features
            )[0][1]

            # ----------------------------------------
            # Determine Risk Level
            # ----------------------------------------

            if probability < 0.30:

                risk_level = "Low"

            elif probability < 0.70:

                risk_level = "Moderate"
            else:

                risk_level = "High"

            logging.info(
                f"Prediction : {prediction}"
            )

            logging.info(
                f"Probability : {probability:.4f}"
            )

            logging.info(
                f"Risk Level : {risk_level}"
            )

            return {
        "prediction": prediction,
        "probability": probability,
            "risk_level": risk_level
            }
        except Exception as e:

            logging.error(
                "Error occurred during prediction."
            )

            raise CustomException(
                e,
                sys
            )

# ============================================================
# Custom Data Class
# ============================================================

class CustomData:
    """
    This class prepares the user input for prediction.

    Workflow
    --------
    1. Load the default values saved during training.
    2. Replace the default values with user inputs.
    3. Return a DataFrame containing all model features.

    This ensures the prediction model always receives the
    complete feature set that it was trained on.
    """

    def __init__(self, user_inputs: dict):
        """
        Parameters
        ----------
        user_inputs : dict

            Dictionary containing the values entered by
            the user from the Streamlit application.

            Example

            {

                "Age":70,

                "Gender":1,

                "BMI":28.5

            }
        """

        self.user_inputs = user_inputs

        self.config = PredictionPipelineConfig()


    def get_data_as_dataframe(self):
        """
        Convert the user inputs into a DataFrame.

        Returns
        -------
        pandas.DataFrame

            DataFrame containing all model features.
        """

        try:

            logging.info(
                "Preparing prediction dataframe."
            )

           
            

            # ----------------------------------------
            # Convert to DataFrame
            # ----------------------------------------

            prediction_df = pd.DataFrame([self.user_inputs])

            logging.info(
                "Prediction dataframe created successfully."
            )

            logging.info(
                f"Prediction Data Shape : {prediction_df.shape}"
            )

            return prediction_df

        except Exception as e:

            logging.error(
                "Error occurred while preparing prediction dataframe."
            )

            raise CustomException(
                e,
                sys
            )

# ============================================================
# Test Prediction Pipeline
# ============================================================

if __name__ == "__main__":

    try:

        logging.info(
            "Testing Prediction Pipeline."
        )

        # ----------------------------------------
        # Example User Inputs
        # ----------------------------------------

        user_inputs = {

            "Age": 72,

            "Gender": 1,

            "Ethnicity": 0,

            "EducationLevel": 2,

            "BMI": 27.5,

            "AlcoholConsumption": 5.0,

            "PhysicalActivity": 3.0,

            "DietQuality": 7.0,

            "SleepQuality": 6.5,

            "SystolicBP": 135,

            "DiastolicBP": 85,

            "CholesterolTotal": 205,

            "CholesterolLDL": 130,

            "CholesterolHDL": 55,

            "CholesterolTriglycerides": 165,

            "MMSE": 18,

            "FunctionalAssessment": 6,

            "ADL": 5,

            "BehavioralProblems": 1,

            "MemoryComplaints": 1

        }

        # ----------------------------------------
        # Create DataFrame
        # ----------------------------------------

        custom_data = CustomData(
            user_inputs=user_inputs
        )

        prediction_df = custom_data.get_data_as_dataframe()

        logging.info(
            "Prediction DataFrame created successfully."
        )

        print("\nPrediction DataFrame\n")

        print(prediction_df)

        # ----------------------------------------
        # Predict
        # ----------------------------------------

        pipeline = PredictPipeline()

        prediction, probability, risk_level = pipeline.predict(
            prediction_df
        )

        print("\nPrediction Results\n")

        print(f"Prediction   : {prediction}")

        print(f"Probability  : {probability:.4f}")

        print(f"Risk Level   : {risk_level}")

        logging.info(
            "Prediction pipeline tested successfully."
        )

    except Exception as e:

        logging.error(
            "Prediction pipeline testing failed."
        )

        raise CustomException(
            e,
            sys
        )



      
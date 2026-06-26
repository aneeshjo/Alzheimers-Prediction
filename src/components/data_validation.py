# Used for file path operations
import os

# Used for system exception handling
import sys

# Used for reading datasets
import pandas as pd

# Custom Exception
from src.exception import CustomException

# Custom Logger
from src.logger import logging


class DataValidation:
    """
    Responsible for validating the dataset before
    sending it to the transformation pipeline.
    """

    def __init__(self):
        pass

    def validate_dataset(self, file_path):
        """
        Validates the dataset.

        Parameters
        ----------
        file_path : str
            Path of the dataset

        Returns
        -------
        bool
            True if validation succeeds
        """

        try:

            logging.info("Starting Data Validation")
            
            # Read dataset
            df = pd.read_csv(file_path)

            logging.info(
                f"Dataset loaded successfully with shape {df.shape}"
            )

            # ===============================
            # Check 1 : Dataset Empty
            # ===============================

            if df.empty:
                raise Exception("Dataset is empty")

            logging.info("Dataset is not empty")

            # ===============================
            # Check 2 : Required Columns
            # ===============================

            required_columns = [
                'PatientID',
                'Age',
                'Gender',
                'Ethnicity',
                'EducationLevel',
                'BMI',
                'Smoking',
                'AlcoholConsumption',
                'PhysicalActivity',
                'DietQuality',
                'SleepQuality',
                'FamilyHistoryAlzheimers',
                'CardiovascularDisease',
                'Diabetes',
                'Depression',
                'HeadInjury',
                'Hypertension',
                'SystolicBP',
                'DiastolicBP',
                'CholesterolTotal',
                'CholesterolLDL',
                'CholesterolHDL',
                'CholesterolTriglycerides',
                'MMSE',
                'FunctionalAssessment',
                'MemoryComplaints',
                'BehavioralProblems',
                'ADL',
                'Confusion',
                'Disorientation',
                'PersonalityChanges',
                'DifficultyCompletingTasks',
                'Forgetfulness',
                'Diagnosis',
                'DoctorInCharge'
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if len(missing_columns) > 0:

                raise Exception(
                    f"Missing Columns Found: {missing_columns}"
                )

            logging.info(
                "All required columns are present"
            )

            # ===============================
            # Check 3 : Target Column
            # ===============================

            if "Diagnosis" not in df.columns:

                raise Exception(
                    "Target column Diagnosis is missing"
                )

            logging.info(
                "Target column found"
            )

            # ===============================
            # Check 4 : Missing Values
            # ===============================

            missing_values = df.isnull().sum().sum()

            logging.info(
                f"Total Missing Values: {missing_values}"
            )

            # ===============================
            # Check 5 : Duplicate Rows
            # ===============================

            duplicate_rows = df.duplicated().sum()

            logging.info(
                f"Duplicate Rows Found: {duplicate_rows}"
            )

            # ===============================
            # Check 6 : Age Validation
            # ===============================

            if (df["Age"] <= 0).any():

                raise Exception(
                    "Invalid Age values found"
                )

            logging.info(
                "Age validation passed"
            )

            # ===============================
            # Check 7 : BMI Validation
            # ===============================

            if (df["BMI"] <= 0).any():

                raise Exception(
                    "Invalid BMI values found"
                )

            logging.info(
                "BMI validation passed"
            )

            # ===============================
            # Check 8 : Diagnosis Validation
            # ===============================

            valid_target_values = {0, 1}

            if not set(df["Diagnosis"].unique()).issubset(
                valid_target_values
            ):

                raise Exception(
                    "Diagnosis contains invalid values"
                )

            logging.info(
                "Diagnosis validation passed"
            )

            logging.info(
                "Data Validation Completed Successfully"
            )

            return True

        except Exception as e:

            logging.error(
                "Error occurred during Data Validation"
            )

            raise CustomException(e, sys)


# For testing the file independently
if __name__ == "__main__":

    validator = DataValidation()

    validation_status = validator.validate_dataset(
        file_path="notebooks\\data\\alzheimers_disease_data.csv"
    )

    print(
        f"Validation Status : {validation_status}"
    )
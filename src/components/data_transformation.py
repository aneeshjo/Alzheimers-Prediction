import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    It defines the path to save the preprocessor object.
    """
    transformation_artifacts_dir: str = os.path.join(
        "artifacts",
        "data_transformation"
    )
    preprocessor_obj_file_path= os.path.join(transformation_artifacts_dir,"preprocesor.pkl")

class DataTransformation:
    """
    Class responsible for data transformation.
    It creates a preprocessing pipeline for both numerical and categorical features,
    and saves the preprocessor object to a file.
    """


    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_path):
        """
        Creates and returns the preprocessing pipeline.

        Parameters
        ----------
        train_path : str
            Path of the training dataset.

        Returns
        -------
        ColumnTransformer
            Preprocessing object containing all transformation pipelines.
        """

        try:

            logging.info("Starting Data Transformation Pipeline Creation.")

            # ==============================
            # Load Training Dataset
            # ==============================

            logging.info("Loading training dataset.")

            train_df = pd.read_csv(train_path)

            logging.info(f"Training dataset loaded successfully with shape {train_df.shape}")

            # ==============================
            # Remove unnecessary columns
            # ==============================

            logging.info("Dropping unnecessary columns.")

            DROP_COLUMNS = [
                "PatientID",
                "DoctorInCharge",
                "Diagnosis"
            ]

            X = train_df.drop(
                columns=DROP_COLUMNS,
                errors="ignore"
            )

            # ==============================
            # Automatically identify feature types
            # ==============================

            logging.info("Identifying feature types.")

            # Continuous numerical columns (float type)
            continuous_columns = X.select_dtypes(
                include=["float64"]
            ).columns.tolist()

            # Integer columns
            integer_columns = X.select_dtypes(
                include=["int64"]
            ).columns.tolist()

            # Binary columns (0/1)
            binary_columns = [
                col
                for col in integer_columns
                if X[col].nunique() == 2
            ]

            # Known categorical columns
            categorical_columns = [
                "Ethnicity",
                "EducationLevel"
            ]

            # Remaining integer columns are treated as numerical
            numerical_integer_columns = [

                col

                for col in integer_columns

                if col not in binary_columns
                and col not in categorical_columns

            ]

            # Combine all numerical columns
            numerical_columns = (
                continuous_columns +
                numerical_integer_columns
            )

            # ==============================
            # Log detected feature types
            # ==============================

            logging.info(f"Continuous Columns : {continuous_columns}")

            logging.info(f"Numerical Integer Columns : {numerical_integer_columns}")

            logging.info(f"Binary Columns : {binary_columns}")

            logging.info(f"Categorical Columns : {categorical_columns}")

            # ==============================
            # Numerical Pipeline
            # ==============================

            logging.info("Creating numerical preprocessing pipeline.")

            num_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )

                ]

            )

            # ==============================
            # Categorical Pipeline
            # ==============================

            logging.info("Creating categorical preprocessing pipeline.")

            cat_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    )

                ]

            )

            # ==============================
            # Binary Pipeline
            # ==============================

            logging.info("Creating binary preprocessing pipeline.")

            binary_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    )

                ]

            )

            # ==============================
            # Create Column Transformer
            # ==============================

            logging.info("Creating ColumnTransformer.")

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "Numerical Pipeline",
                        num_pipeline,
                        numerical_columns
                    ),

                    (
                        "Categorical Pipeline",
                        cat_pipeline,
                        categorical_columns
                    ),

                    (
                        "Binary Pipeline",
                        binary_pipeline,
                        binary_columns
                    )

                ]

            )

            logging.info("Data Transformation Pipeline Created Successfully.")

            return preprocessor

        except Exception as e:

            logging.error("Error occurred while creating preprocessing pipeline.")

            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process.
        
        Parameters:
        - train_path: str, path to the training data file
        - test_path: str, path to the testing data file
        
        Returns:
        - Tuple containing:
            - preprocessor object
            - transformed training features
            - transformed testing features
            - training target variable
            - testing target variable
        """
        try:
            logging.info("Starting data transformation process.")
            
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Define the target variable and drop it from the feature set
            target_column_name = 'Diagnosis'

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object(train_path=train_path)

            # Fit and transform the training features, and transform the testing features
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed successfully.")

            transformed_train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            transformed_test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Transformed training and testing arrays created successfully.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )


            return (
                transformed_train_arr,
                transformed_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)


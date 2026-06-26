# Some utility functions for the project
import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

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


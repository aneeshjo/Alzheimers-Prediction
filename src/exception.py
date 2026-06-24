# Some utility functions for the project
import sys


def error_message_detail(error,error_detail:sys):
     """
    Extracts detailed information about an exception:
    - File name where the error occurred
    - Line number of the error
    - Actual error message
    """
     
    # Get the exception type, value, and traceback
     __,_,exc_tb = error_detail.exc_info()

    # Get the file name from the traceback
     file_name = exc_tb.tb_frame.f_code.co_filename

    # Get the line number from the traceback
     line_number = exc_tb.tb_lineno

    # Create a detailed error message
     # Create a formatted error message
     error_message = (
        f"Error occurred in script: {file_name} "
        f"at line number: {line_number} "
        f"with error message: {str(error)}"
    )
    
     return error_message


class CustomException(Exception):
    """
    Custom exception class that extends Python's built-in Exception.
    It automatically formats the error message using error_message_details().
    """
    def __init__(self, error, error_detail:sys):
        # Call the parent class constructor with the formatted error message
        super().__init__(error_message_detail(error, error_detail))

        # Create a detailed error message using our helper function
        self.error_message=error_message_detail(error, error_detail)

    def __str__(self):
        # When the exception is printed, return the formatted message
        return self.error_message
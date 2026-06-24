# Import Python's built-in logging module
# Used to track program execution, warnings, errors, and debugging information
import logging as logger

# Import os module
# Used for creating folders and working with file paths
import os

# Import datetime module
# Used to generate unique log file names with current date and time
from datetime import datetime

# Create a log file name using current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the path where logs will be stored

logs_path = os.path.join(
    os.getcwd(),     # Current project directory
    "logs",          # Logs folder
    
)

# Create the folder if it doesn't already exist
# exist_ok=True prevents an error if the folder already exists
os.makedirs(logs_path, exist_ok=True)

# Create the complete path of the log file
# logs\06_24_2026_10_30_15.log\06_24_2026_10_30_15.log
LOG_FILE_PATH = os.path.join(
    logs_path,
    LOG_FILE
)

# Configure logging settings
logger.basicConfig(

    # Location where logs will be stored
    filename=LOG_FILE_PATH,

    # Format of each log message
    format="[%(asctime)s] %(levelname)s - %(message)s",

    # Record INFO, WARNING, ERROR and CRITICAL messages
    level=logger.INFO
)

# This block executes only when this file is run directly
# Useful for testing the logger
# if __name__ == "__main__":

#     # Log an informational message
#     logger.info("Logging has started.")

#     # Log another informational message
#     logger.info("This is an info message.")

#     # Log a warning message
#     logger.warning("This is a warning message.")

#     # Log an error message
#     logger.error("This is an error message.")

#     # Log a critical/fatal message
#     logger.critical("This is a critical message.")
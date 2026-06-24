# Import setup tools used to create a Python package
from setuptools import setup, find_packages

# This is used to identify editable installation
# Example: pip install -e .
HYPHEN_E_DOT = "-e ."


# Function to read all libraries from requirements.txt
def get_requirements(filename):

    # Open the requirements file
    with open(filename) as file_object:

        # Read each line, remove spaces/newlines,
        # ignore empty lines and comments
        requirements = [
            line.strip()
            for line in file_object
            if line.strip() and not line.startswith('#')
        ]

    # Remove "-e ." because it is not a package dependency
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    # Return the final list of packages
    return requirements


# Project/package configuration
setup(

    # Name of the project
    name='Alzheimers_Prediction',

    # Version of the project
    version='0.0.1',

    # Author name
    author="Aneesh Jose",

    # Author email
    author_email="aneeshjose012@gmail.com",

    # Short description of the project
    description="Machine Learning based Alzheimer's Disease Prediction System",

    # Automatically find all Python packages/folders
    packages=find_packages(),

    # Install all libraries listed in requirements.txt
    install_requires=get_requirements("requirements.txt")
)
import os
import logging

from pathlib import Path

# Configure logging to include timestamps for better traceability
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of file paths to be created.
# Directories auto-created if they don't exist.
list_of_files = [
    "src/ml_algorithms/__init__.py",
    "src/ml_algorithms/kmeans.py",
    "pyproject.toml",
    "requirements/models.txt",
    "requirements/tests.txt",
    "test/kmeans.py",
]

# Iterate through each file path
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, _ = os.path.split(filepath)
    
    # Create the directory if it doesn't already exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir}')
    
    # Create the file if it doesn't exist or is currently empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")

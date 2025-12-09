import logging
import os
from datetime import datetime


print(os.getlogin())
LOG_DIR = f"C:\\Users\\{os.getlogin()}\\Documents\\logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"clinicalBERT_log_{timestamp}.log")

def get_logger(name="app_logger"):
    
    """
   Standar logger format.
   Find the logs on the route C:\\Users\\YOUR_USERNAME\\Documents\\logs\\

        Args:
            name: Name of the logger.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    
    logger.propagate = False

    logger.info(f"The logger has been initialized {name}")
    return logger

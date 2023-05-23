"""making logging for all the scripts, the script is started by chat.openai"""

import logging


def setup_logger():
    """
    Set up and configure the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger instance with the module name
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)

    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

"""making logging for all the scripts, the script is started by chat.openai"""

import os
import re
import logging
import datetime


# check if the log file is exist rename the new file
log_files = [file for file in os.listdir('.') if
             os.path.isfile(file) and file.startswith('log') and
             re.match(r'log.\d+', file)]

COUNT = 1
if len(log_files) > 0:
    COUNT = max(int(re.search(r'log.(\d+)', file).group(1)) for
                file in log_files) + 1 if log_files else 1
log_file = f"log.{COUNT}"

with open(log_file, 'w', encoding='utf-8') as f_w:
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    f_w.write(f'{current_datetime}\n')
    f_w.write('\n')


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
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Define the log message format
    formatter = logging.Formatter(
        '%(module)s: (%(filename)s)\n\t - %(message)s\n')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

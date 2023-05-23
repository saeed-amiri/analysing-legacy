"""making logging for all the scripts, the script is started by chat.openai"""

import os
import re
import logging
import datetime
from colors_text import TextColor as bcolors

def check_log_file(log_name: str  # name of the asked logfile
                   ) -> str:
    """check if the log file is exist rename the new file"""
    # Get a list of log files in the directory
    log_files = [file for file in os.listdir('.') if
                 re.match(fr'{log_name}.\d+', file)]

    # Find the maximum count of the log files and increment it by 1
    pattern = fr'{log_name}\.(\d+)'
    count = max(int(re.search(pattern, file).group(1)) for
                file in log_files) + 1 if log_files else 1

    # Create the new log file name
    new_log_file = fr'{log_name}.{ count}'
    print(f'{bcolors.OKBLUE}{__name__}: The log file '
          f'`{new_log_file}` is prepared{bcolors.ENDC}')
    return new_log_file


def write_header(log_file: str  # name of the asked logfile
                 ) -> None:
    """write the header of the file"""
    with open(log_file, 'w', encoding='utf-8') as f_w:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        f_w.write(f'{formatted_datetime}\n')
        f_w.write(f'{os.getcwd()}\n')
        f_w.write('\n')


def setup_logger(log_name):
    """
    Set up and configure the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger instance with the module name
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # check if the log file is exist rename the new file
    log_file: str = check_log_file(log_name)

    # Write the header of the log file
    write_header(log_file)

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

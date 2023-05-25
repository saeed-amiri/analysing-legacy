"""tools used in multiple scripts"""
import os
import sys
import logger
from colors_text import TextColor as bcolors


def check_file_exist(fname: str,  # Name of the file to check
                     log: logger.logging.Logger  # log the error
                     ) -> None:
    """check if the file exist, other wise exit"""
    if not os.path.exists(fname):
        log.error(f'Error! `{fname}` dose not exist.')
        sys.exit(f'{bcolors.FAIL}{__name__}: '
                 f'(Error! `{fname}` dose not '
                 f'exist \n{bcolors.ENDC}')

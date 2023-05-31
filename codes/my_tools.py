"""tools used in multiple scripts"""
import os
import re
import sys
import typing
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
    else:
        log.info(f'reading: `{fname}`')


def drop_string(input_string: str,
                string_to_drop: str
                ) -> str:
    output_string = input_string.replace(string_to_drop, "")
    return output_string


def extract_string(input_string: str) -> list[typing.Any]:
    pattern = r'"(.*?)"'
    matches = re.findall(pattern, input_string)
    return matches

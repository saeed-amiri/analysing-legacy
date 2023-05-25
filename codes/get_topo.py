"""this script reads the "topol.top" file to get the paths of the itp
files and determine the number of each residue in the system; it is
necessary to refer to the topology file. Remember that since I moved
one directory, the "itp" files will be located one back."""

import sys
import logger
import my_tools
from colors_text import TextColor as bcolors


class ReadTop:
    """read the topology file;
    It should named: topol.top"""
    def __init__(self) -> None:
        self.fanme: str = 'topol.top'
        my_tools.check_file_exist(self.fanme, logger.setup_logger('test_log'))


if __name__ == "__main__":
    top = ReadTop()

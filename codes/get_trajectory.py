"""The script is designed to read the trajectory files and extract
essential information from them, including the number of molecules,
frames, and atoms. """

import os
import sys


class GetInfo:
    """read trr files and return info from it"""
    def __init__(self,
                 fname: str  # Name of the trr file
                 ) -> None:
        if not self.path_esist(fname):
            sys.exit('No such file')
        self.trr: str = fname
        self.__get_gro()

    def __get_gro(self) -> str:
        """return the name of the gro file and check if it exist"""
        tmp: str = self.trr.split('.')[0]
        gro_file: str = f'{tmp}.gro'
        if self.path_esist(gro_file):
            print('gro dosenot exist')
        return gro_file

    def path_esist(self,
                   fname: str  # Name of the file to check
                   ) -> bool:
        """check if the file exist"""
        return os.path.exists(fname)


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1])

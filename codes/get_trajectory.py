"""The script is designed to read the trajectory files and extract
essential information from them, including the number of molecules,
frames, and atoms. """

import os
import sys
import logger
import MDAnalysis as mda
from colors_text import TextColor as bcolors


class GetInfo:
    """read trr files and return info from it"""
    def __init__(self,
                 fname: str,  # Name of the trr file
                 log: logger.logging.Logger  # Name of the log file
                 ) -> None:
        if not self.path_esist(fname):
            sys.exit('No such file')
        self.trr: str = fname
        self.gro: str = self.__get_gro(log)  # Toopology file
        self.u_traj: mda.Universe = self.read_traj(log)
        self.__get_info(log)

    def __get_info(self,
                   log: logger.logging.Logger
                   ) -> mda.Universe:
        """get all the info from the input files"""
        self.__get_residues()

    def __get_residues(self) -> dict:
        """get the all the residues in the dictionary"""
        # Create a dictionary to store the residue indices
        residue_indices: dict[str, list[int]] = {}

        # Iterate over the residues
        for residue in self.u_traj.residues:
            residue_name = residue.resname

            # Get the index of the current residue
            residue_index = residue.resindex

            # Add the index to the dictionary
            if residue_name not in residue_indices:
                residue_indices[residue_name] = []
            residue_indices[residue_name].append(residue_index)
        return residue_indices

    def read_traj(self,
                  log: logger.logging.Logger
                  ) -> mda.Universe:
        """read traj and topology file"""
        log.info(f'Trajectory file `{self.trr}` and topology file'
                 f' `{self.gro}` are read.')
        return mda.Universe(self.gro, self.trr)

    def __get_gro(self,
                  log: logger.logging.Logger  # Name of the log file
                  ) -> str:
        """return the name of the gro file and check if it exist"""
        tmp: str = self.trr.split('.')[0]
        gro_file: str = f'{tmp}.gro'
        if not self.path_esist(gro_file):
            log.error(f'Error! `{gro_file}` dose not exist.')
            sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}: '
                     f'({self.__module__})\n Error! `{gro_file}` dose not '
                     f'exist \n{bcolors.ENDC}')
        return gro_file

    def path_esist(self,
                   fname: str  # Name of the file to check
                   ) -> bool:
        """check if the file exist"""
        return os.path.exists(fname)


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('test_log'))

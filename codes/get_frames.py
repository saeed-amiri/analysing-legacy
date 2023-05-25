"""this script uses the data from get_trajectory.py and returns the
residues' positions in the 3d matrix. Then it will; it will get the
center of mass of each residue at each timestep.
The scripts initiated by chatGpt
"""

import sys
import numpy as np
import logger
import static_info as stinfo
from get_trajectory import GetInfo


class ResiduePositions:
    """getting posotions of the """
    def __init__(self,
                 trr_info: GetInfo  # All the info from trr and gro files
                 ):
        self.info: GetInfo = trr_info
        self.__allocate()
        # self.residue_positions = self.get_residue_positions()

    def __allocate(self) -> None:
        """allocate arraies for saving all the info"""
        frames: int = self.info.num_dict['n_frames']
        unique_residues: int = len(stinfo.atoms_num.keys())
        # update the residues index to get the NP: APT_COR
        self.__get_np_index()
        print(frames, unique_residues, len(self.info.residues_indx['ODN']))
        print(self.info.residues_indx.keys())

    def __get_np_index(self) -> dict[str, list[int]]:
        residue_indx: dict[str, list[int]] = {}
        npi: str = stinfo.np_info['np_residues'][0]
        npj: str = stinfo.np_info['np_residues'][1]
        for k in stinfo.atoms_num.keys():
            if k in stinfo.np_info['solution_residues']:
                residue_indx[k] = self.info.residues_indx[k].copy()
        np_name: str = npi + npj
        residue_indx[np_name] = self.info.residues_indx[npi].copy() + \
            self.info.residues_indx[npi].copy()
        return residue_indx


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('get_frames_log'))
    positions = ResiduePositions(trr)

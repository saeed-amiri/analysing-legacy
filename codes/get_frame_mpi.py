"""
Read the trajectory file and return the residues' center of mass. There
are a few things that have to be considered:
    - Save the time of each step to ensure the correct setting.
    - Set an index for each residue's type after its COM for later data
      following.
    - Save the COM of the amino group of the ODN.
In the final array, rows indicate the timeframe, and columns show the
center of mass of the residues.

n_residues: number of the residues
n_ODA: number oda residues
NP_com: Center of mass of the nanoparticle
Shape of the array will be:

number of the columns:
timeframe + NP_com + n_residues: xyz,type + n_oda * xyz,type
     1    +   3    + n_residues * 4       + n_oda * 4

and  the number of rows will be the number of timeframes

"""


import sys
import json
import numpy as np
import logger
import static_info as stinfo
import get_topo as topo
from get_trajectory import GetInfo


class GetResiduesPosition:
    """
    Get the positions of the residues, based on the description on the
    doc.
    """

    info_msg: str = 'Messages:\n'  # To log
    # The following will set in _initiate_reading
    top: topo.ReadTop  # Topology file
    trr_info: GetInfo  # All the info from trajectory

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger
                 ) -> None:
        self._initiate_reading(fname, log)
        self._initiate_data()

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: logger.logging.Logger
                          ) -> None:
        """
        Call the other modules and read the files
        """
        self.top = topo.ReadTop(log)
        self.trr_info = GetInfo(fname, log=log)

    def _initiate_data(self) -> None:
        """
        Initiate setting data to get the COM of the residues
        MDAnalysis cant manage the indicies correctly. So, I have to
        save and set the indicies for NP (COR & APT) and solution
        separately.
        """
        sol_res_tmp: dict[str, list[int]] = \
            self.get_residues(stinfo.np_info["solution_residues"])
        np_res_tmp: dict[str, list[int]] = \
            self.get_residues(stinfo.np_info["np_residues"])
        sol_res = self.set_residues_index(sol_res_tmp)
        np_res = self.set_residues_index(np_res_tmp)

    @staticmethod
    def set_residues_index(all_res_tmp: dict[str, list[int]]  # Name&index
                           ) -> dict[int, int]:
        """set the type of the each residue as an index"""
        all_res_dict: dict[int, int] = {}  # All the residues with int type
        for k, val in all_res_tmp.items():
            for res in val:
                all_res_dict[res] = stinfo.reidues_id[k]
        return all_res_dict

    def get_residues(self,
                     res_name: list[str]  # Name of the residues
                     ) -> dict[str, list[int]]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        self.info_msg += '\tGetting the residues:\n'
        self.info_msg += f'\t{json.dumps(res_name, indent=8)}'
        all_res_dict: dict[str, list[int]] = {}  # All the residues in solution
        for k, val in self.trr_info.residues_indx.items():
            if k in res_name:
                all_res_dict[k] = val
        return all_res_dict


if __name__ == '__main__':
    GetResiduesPosition(fname=sys.argv[1],
                        log=logger.setup_logger('get_frames_mpi_log'))
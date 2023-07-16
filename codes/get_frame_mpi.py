#!/usr/bin/env python3
"""
Read the trajectory file and return the residues' center of mass. There
are a few things that have to be considered:
    - Save the time of each step to ensure the correct setting.
    - Set an index for each residue's type after its COM for later data
      following.
    - Save the COM of the amino group of the ODN.
In the final array, rows indicate the timeframe, and columns show the
center of mass of the residues.

number of row will be"
    number of frames + 1
    The extra row is for the type of the residue


number of the columns:
n_residues: number of the residues
n_ODA: number oda residues
NP_com: Center of mass of the nanoparticle
than:
timeframe + NP_com + n_residues:  xyz + n_oda * xyz
     1    +   3    +  n_residues * 3  +  n_oda * 3

"""


import sys
from mpi4py import MPI
import logger
import static_info as stinfo
import get_topo as topo
from get_trajectory import GetInfo
from colors_text import TextColor as bcolors


class GetResidues:
    """
    Get the residues, based on the description on the doc.
    """

    info_msg: str = 'Messages:\n'  # To log
    # The following will set in _initiate_reading
    top: topo.ReadTop  # Topology file
    trr_info: GetInfo  # All the info from trajectory
    sol_res: dict[int, int]  # Residues with their type as an integer
    np_res: dict[int, int]  # Residues with their type as an integer

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger
                 ) -> None:
        self._initiate_reading(fname, log)
        self.sol_res, self. np_res = self._initiate_data()
        self.__write_msg(log)

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: logger.logging.Logger
                          ) -> None:
        """
        Call the other modules and read the files
        Top contains info from forcefield parameters files, the numbers
        of the residues.
        ttr_info contains trajectory read by MDAnalysis
        """
        self.top = topo.ReadTop(log)
        self.trr_info = GetInfo(fname, log=log)

    def _initiate_data(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        Initialize setting data to obtain the center of mass of
        residues. MDAnalysis fails to manage indices, so NP (COR & APT)
        and solution indices are saved and set separately.

        Algorithm:
            Retrieve all residues for nanoparticles and solution, then
            set an index for each based on type defined in static_info.

            Returns:
            Dictionaries for each set with residue index as keys and
            the reisdues' type as values.

        """
        sol_res_tmp: dict[str, list[int]] = \
            self.get_residues(stinfo.np_info["solution_residues"])
        np_res_tmp: dict[str, list[int]] = \
            self.get_residues(stinfo.np_info["np_residues"])
        sol_res = self.set_residues_index(sol_res_tmp)
        np_res = self.set_residues_index(np_res_tmp)
        return sol_res, np_res

    @staticmethod
    def set_residues_index(all_res_tmp: dict[str, list[int]]  # Name&index
                           ) -> dict[int, int]:
        """set the type of the each residue as an index"""
        return \
            {res: stinfo.reidues_id[k] for k, val in all_res_tmp.items()
             for res in val}

    def get_residues(self,
                     res_name: list[str]  # Name of the residues
                     ) -> dict[str, list[int]]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        self.info_msg += '\tGetting the residues:\n'
        all_res_dict: dict[str, list[int]] = {}  # All the residues in solution
        all_res_dict = \
            {k: val for k, val in self.trr_info.residues_indx.items()
             if k in res_name}
        self.info_msg += \
            (f'\tThe number of the read residues for {res_name} is:\n'
             f'\t\t`{sum(len(lst) for lst in all_res_dict.values())}`\n')
        return all_res_dict

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetResidues.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class CalculateCom:
    """
    Calculating the center of mass (COM) of the residues.
    input:
        The objects of the GetResidues class
    Output:
        An array contains infos defined in the script's doc
    """

    info_msg: str = 'Messages:\n'  # To log

    def __init__(self,
                 fname: str,  # Name of the trajectory files
                 log: logger.logging.Logger
                 ) -> None:
        self.get_residues = GetResidues(fname, log)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


if __name__ == '__main__':
    CalculateCom(fname=sys.argv[1],
                 log=logger.setup_logger('get_frames_mpi_log'))

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
import logger
import MDAnalysis as mda
import static_info as stinfo
import my_tools
import get_topo as topo
from get_trajectory import GetInfo


class GetResiduesPosition:
    """
    Get the positions of the residues, based on the description on the
    doc.
    """

    info_msg: str = 'Messages:\n'  # To log

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger
                 ) -> None:
        self._initiate_reading(fname, log)

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: logger.logging.Logger
                          ) -> None:
        """
        Call the other modules and read the files
        """
        self.top = topo.ReadTop(log)
        self.trr_info = GetInfo(fname, log=log)


if __name__ == '__main__':
    GetResiduesPosition(fname=sys.argv[1],
                        log=logger.setup_logger('get_frames_mpi_log'))

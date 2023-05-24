"""This script is designed for the analysis of simulation trajectories.
It primarily requires the trajectory files, along with the corresponding
topology and index files. Additionally, it can utilize the gro file,
which represents the final state of the system. These input files enable
the script to extract relevant data and perform various analyses on the
simulation trajectories."""


import sys
import logger
import get_trajectory as traj


if __name__ == '__main__':
    trr = traj.GetInfo(sys.argv[1], log=logger.setup_logger('test_log'))

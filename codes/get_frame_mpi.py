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
import typing
from mpi4py import MPI
import numpy as np
import logger
import static_info as stinfo
import get_topo as topo
from get_trajectory import GetInfo
from colors_text import TextColor as bcolors


class GetResidues:
    """
    Get the residues, based on the description on the doc.
    """

    info_msg: str = 'Messages from GetResidues:\n'  # To log
    # The following will set in _initiate_reading
    top: topo.ReadTop  # Topology file
    trr_info: GetInfo  # All the info from trajectory
    sol_res: dict[int, int]  # Residues with their type as an integer
    np_res: dict[int, int]  # Residues with their type as an integer

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: typing.Union[logger.logging.Logger, None]
                 ) -> None:
        self._initiate_reading(fname, log)
        self.sol_res, self. np_res = self._initiate_data()
        self.__write_msg(log)

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: typing.Union[logger.logging.Logger, None]
                          ) -> None:
        """
        Call the other modules and read the files
        Top contains info from forcefield parameters files, the numbers
        of the residues.
        ttr_info contains trajectory read by MDAnalysis
        """
        if log is not None:
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
                    log: typing.Union[logger.logging.Logger, None]  # To log
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetResidues.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        if log is not None:
            log.info(self.info_msg)


class CalculateCom:
    """
    Calculating the center of mass (COM) of the residues.
    input:
        The objects of the GetResidues class
    Output:
        An array contains infos defined in the script's doc
    """

    info_msg: str = 'Messages from CalculateCom:\n'  # To log
    get_residues: typing.Union[GetResidues, None]  # Type of the info

    def __init__(self,
                 fname: str,  # Name of the trajectory files
                 log: typing.Union[logger.logging.Logger, None]
                 ) -> None:
        self._initiate_data(fname, log)
        self._initiate_calc()
        self.__write_msg(log)

    def _initiate_data(self,
                       fname: str,  # Name of the trajectory files
                       log: typing.Union[logger.logging.Logger, None]
                       ) -> None:
        """
        This function Call GetResidues class and get the data from it.
        """
        if RANK == 0:
            self.get_residues = GetResidues(fname, log)
            self.n_frames = self.get_residues.trr_info.num_dict['n_frames']
            self.info_msg += (f'\tNumber of processes is: `{SIZE}`\n'
                              f'\tNumber of frames is: `{self.n_frames}`\n')
        else:
            self.get_residues = None

    def _initiate_calc(self) -> None:
        """
        First divide the list, than brodcast between processes.
        Get the lists the list contains timesteps.
        The number of sublist is equal to number of cores, than each
        sublist will be send to one core.
        The total trajectory will br brodcast to all the processors

        Args:
            None

        Returns:
            None

        Notes:
            - The `n_frames` should be equal or bigger than n_process,
              otherwise it will reduced to n_frames
            - u_traj: <class 'MDAnalysis.coordinates.TRR.TRRReader'>
            - chunk_tstep: typing.Union[list[list[np.ndarray]], None]
        """
        if RANK == 0:
            data: np.ndarray = np.arange(self.n_frames)
            chunk_tstep = self.get_chunk_lists(data)
            np_res_ind = self.get_np_residues()
            if self.get_residues is not None:
                u_traj = self.get_residues.trr_info.u_traj
        else:
            chunk_tstep = None
            np_res_ind = None
            u_traj = None
        # Setting the type
        chunk_tstep = typing.cast(typing.List[typing.Any], chunk_tstep)
        # Broadcast and scatter all the data
        chunk_tstep = COMM.scatter(chunk_tstep, root=0)
        np_res_ind = COMM.bcast(np_res_ind, root=0)
        u_traj = COMM.bcast(u_traj, root=0)

        self.process_trj(RANK, chunk_tstep, u_traj, np_res_ind)

    def process_trj(self,
                    i_rank: int,  # Rank of the processor
                    chunk_tstep,  # Frames' ind
                    u_traj,  # Trajectory
                    np_res_ind: typing.Union[list[int], None]  # NP residues id
                    ) -> None:
        """Get atoms in the timestep"""
        self.get_com(RANK, chunk_tstep)
        if chunk_tstep is not None:
            for i in chunk_tstep:
                frame = u_traj.trajectory[i]
                # Process the frame as needed
                np_com = self.get_np_com(frame.positions, np_res_ind, u_traj)
                print(i_rank, i, np_com, end=', ')

    def get_com(self,
                i_rank: int,  # Rank of the process
                chunk_tstep,  # Index of frames
                ) -> None:
        """Do sth here"""
        print(f'Process {i_rank} has chunk_tstep:', chunk_tstep)

    def get_np_com(self,
                   all_atoms: np.ndarray,  # Atoms positions
                   np_res_ind: typing.Union[list[int], None],  # Residues in NP
                   u_traj
                   ) -> typing.Union[np.ndarray, None]:
        """get the COM for each time step"""
        i_com: list[np.ndarray] = []  # Arrays contains center of masses
        total_mass: float = 0  # Total mass of each residue in the NP
        if np_res_ind is not None:
            for i in np_res_ind:
                com: np.ndarray  # Center of mass of the residue i
                tmp_mass: float  # Mass of the residue
                com, tmp_mass = self.get_np_com_tstep(i, all_atoms, u_traj)
                total_mass += tmp_mass
                i_com.append(com)
                step_com = np.vstack(i_com)
            return np.sum(step_com, axis=0) / total_mass
        return None

    def get_np_com_tstep(self,
                         res_ind: int,  # index of the residue
                         all_atoms: np.ndarray,  # Atoms positions
                         u_traj
                         ) -> tuple[np.ndarray, float]:
        """
        calculate the center of mass of each time step for NP
        """
        i_residue = \
            u_traj.select_atoms(f'resnum {res_ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        tmp_mass = np.sum(atom_masses)
        com = np.average(atom_positions, weights=atom_masses,
                         axis=0) * tmp_mass
        return com, tmp_mass

    def get_np_residues(self) -> list[int]:
        """
        return list of the integer of the residues in the NP
        """
        np_res_ind: list[int] = []  # All the index in the NP
        if self.get_residues is not None:
            for item in stinfo.np_info['np_residues']:
                np_res_ind.extend(
                    self.get_residues.trr_info.residues_indx[item])
        return np_res_ind

    @staticmethod
    def get_chunk_lists(data: np.ndarray  # Range of the time steps
                        ) -> typing.Any:
        """prepare chunk_tstep based on the numbers of frames"""
        # determine the size of each sub-task
        ave, res = divmod(data.size, SIZE)
        counts: list[int]  # Length of each array in the list
        counts = [ave + 1 if p < res else ave for p in range(SIZE)]
        # determine the starting and ending indices of each sub-task
        starts: list[int]  # Start of each list of ranges
        ends: list[int]  # Ends of each list of ranges
        starts = [sum(counts[: p]) for p in range(SIZE)]
        ends = [sum(counts[: p+1]) for p in range(SIZE)]
        # converts data into a list of arrays
        chunk_tstep = [data[starts[p]: ends[p]].astype(np.int32)
                       for p in range(SIZE)]
        return chunk_tstep

    def __write_msg(self,
                    log: typing.Union[logger.logging.Logger, None]  # To log
                    ) -> None:
        """write and log messages"""
        if RANK == 0:
            print(f'{bcolors.OKCYAN}{CalculateCom.__name__}:\n'
                  f'\t{self.info_msg}{bcolors.ENDC}')
            if log is not None:
                log.info(self.info_msg)


if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

    LOG: typing.Union[logger.logging.Logger, None]
    if RANK == 0:
        LOG = logger.setup_logger('get_frames_mpi_log')
    else:
        LOG = None
    CalculateCom(fname=sys.argv[1], log=LOG)

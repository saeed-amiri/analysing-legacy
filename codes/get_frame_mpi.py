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
n_residues: number of the residues in solution, without residues in NP
n_ODA: number oda residues
NP_com: Center of mass of the nanoparticle
than:
timeframe + NP_com + n_residues:  xyz + n_oda * xyz
     1    +   3    +  n_residues * 3  +  n_oda * 3

"""


import sys
import typing
import datetime
import atexit
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
    # The number of resdiues are set in _initiate_data
    nr_sol_res: int  # Number of residues in solution (without NP)
    nr_np_res: int  # Number of residues in NP
    max_res: int  # Maxmum index of the residues in solution
    min_res: int  # Minimum index of the residues in solution

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
        sol_res_tmp: dict[str, list[int]]
        np_res_tmp: dict[str, list[int]]
        sol_res_tmp, self.nr_sol_res, self.max_res, self.min_res = \
            self.get_residues(stinfo.np_info["solution_residues"])
        np_res_tmp, self.nr_np_res, _, _ = \
            self.get_residues(stinfo.np_info["np_residues"])
        sol_res = self.set_residues_index(sol_res_tmp)
        np_res = self.set_residues_index(np_res_tmp)
        return sol_res, np_res

    @staticmethod
    def set_residues_index(all_res_tmp: dict[str, list[int]]  # Name&index
                           ) -> dict[int, int]:
        """set the type of the each residue as an index and number of
        residues"""
        return \
            {res: stinfo.reidues_id[k] for k, val in all_res_tmp.items()
             for res in val}

    def get_residues(self,
                     res_name: list[str]  # Name of the residues
                     ) -> tuple[dict[str, list[int]], int, int, int]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        self.info_msg += '\tGetting the residues:\n'
        all_res_dict: dict[str, list[int]] = {}  # All the residues in solution
        all_res_dict = \
            {k: val for k, val in self.trr_info.residues_indx.items()
             if k in res_name}
        nr_residues = int(sum(len(lst) for lst in all_res_dict.values()))
        max_res = int(max(max(lst) for lst in all_res_dict.values()))
        min_res = int(min(min(lst) for lst in all_res_dict.values()))
        self.info_msg += \
            (f'\tThe number of the read residues for {res_name} is:\n'
             f'\t\t`{nr_residues}`\n'
             f'\t\tThe max & min of indices are: `{max_res}` and'
             f' `{min_res}`\n')
        return all_res_dict, nr_residues, max_res, min_res

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
            sol_residues: typing.Union[dict[str, list[int]], None] = \
                self.get_solution_residues()
            if self.get_residues is not None:
                u_traj = self.get_residues.trr_info.u_traj
                com_arr: typing.Union[np.ndarray, None] = \
                    self.mk_allocation(self.n_frames,
                                       self.get_residues.max_res,
                                       self.get_residues.top.mols_num['ODN'])
                if com_arr is not None:
                    _, com_col = np.shape(com_arr)
        else:
            sol_residues = None
            chunk_tstep = None
            np_res_ind = None
            com_arr = None
            com_col = None
            u_traj = None
        # Setting the type
        chunk_tstep = typing.cast(typing.List[typing.Any], chunk_tstep)
        # Broadcast and scatter all the data
        chunk_tstep = COMM.scatter(chunk_tstep, root=0)
        np_res_ind, com_arr, com_col, u_traj, sol_residues = \
            self.breodcaste_arg(
                np_res_ind, com_arr, com_col, u_traj, sol_residues)

        if chunk_tstep is not None:
            chunk_size = len(chunk_tstep)
        my_data = np.empty((chunk_size, com_col)) if \
            chunk_tstep is not None else None

        my_data = self.process_trj(
                                   chunk_tstep,
                                   u_traj,
                                   np_res_ind,
                                   my_data,
                                   sol_residues
                                   )

        # Gather all the com_arr data to the root process
        if com_arr is not None:
            com_arr_all = COMM.gather(my_data, root=0)
        else:
            com_arr_all = None

        # Combine the gathered com_arr data on the root process
        if RANK == 0 and com_arr_all is not None:
            final_com_arr = np.vstack(tuple(com_arr_all))
            print(final_com_arr)
        # Set the info_msg
        self.get_processes_info(RANK, chunk_tstep)

    @staticmethod
    def breodcaste_arg(*args  # All the things that should be broadcasted
                       ) -> tuple[typing.Any, ...]:
        """
        Broadcasting data
        """
        broad_list: list[typing.Any] = []
        for arg in args:
            broad_list.append(COMM.bcast(arg, root=0))
        return tuple(broad_list)

    def process_trj(self,
                    chunk_tstep,  # Frames' ind
                    u_traj,  # Trajectory
                    np_res_ind: typing.Union[list[int], None],  # NP residue id
                    my_data: typing.Union[np.ndarray, None],  # To save COMs
                    sol_residues: typing.Union[dict[str, list[int]], None]
                    ) -> typing.Union[np.ndarray, None]:
        """Get atoms in the timestep"""
        if chunk_tstep is not None and my_data is not None:
            for row, i in enumerate(chunk_tstep):
                ind = int(i)
                frame = u_traj.trajectory[ind]
                atoms_position: np.ndarray = frame
                np_com = self.get_np_com(atoms_position, np_res_ind, u_traj)
                # Update my_data with ind and np_com values
                my_data[row, 0] = ind
                my_data[row, 1:4] = np_com
                for k, val in sol_residues.items():
                    for item in val:
                        com = self.get_com_all(atoms_position, item)
                        if com is None:
                            continue  # Skip if com is None
                        wrap_com = self.wrap_position(com, frame.dimensions)
                        element = int(item*3) + 1
                        my_data[row][element:element+3] = wrap_com
                        r_idx = stinfo.reidues_id[k]  # Residue idx
                        my_data[-1][element:element+3] = \
                            np.array([[r_idx, r_idx, r_idx]])
            return my_data
        return None

    def get_com_all(self,
                    all_atoms: np.ndarray,  # All the atoms position
                    ind: int  # Index of the residue
                    ) -> typing.Union[np.ndarray, None]:
        """
        return all the residues com
        """
        if self.get_residues is not None:
            i_residue = \
                self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
            atom_indices = i_residue.indices
            atom_positions = all_atoms[atom_indices]
            atom_masses = i_residue.masses
            return np.average(atom_positions, weights=atom_masses, axis=0)
        return None

    def get_processes_info(self,
                           i_rank: int,  # Rank of the process
                           chunk_tstep,  # Index of frames
                           ) -> None:
        """Write info message from each processor"""
        info_msg_local: str = \
            (f'\tProcess {i_rank: 3d} got `{len(chunk_tstep)}` '
             f'timesteps: {chunk_tstep}\n')
        info_msg_all = COMM.gather(info_msg_local, root=0)
        if RANK == 0:
            for msg in info_msg_all:
                self.info_msg += msg

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

    def get_solution_residues(self) -> dict[str, list[int]]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        sol_dict: dict[str, list[int]] = {}  # All the residues in solution
        for k, val in self.get_residues.trr_info.residues_indx.items():
            if k in stinfo.np_info['solution_residues']:
                sol_dict[k] = val
        return sol_dict

    @staticmethod
    def wrap_position(pos: np.ndarray,  # The center of mass
                      vec: np.ndarray  # Box vectors
                      ) -> np.ndarray:
        """
        Wraped the position to the box for unwraped trr.
        """
        if pos is not None:
            for i in range(3):
                pos[i] -= np.floor(pos[i]/vec[i])*vec[i]
            return pos
        return np.zeros(3)

    @staticmethod
    def mk_allocation(n_frames: int,  # Number of frames
                      max_residues: int,  # Max of residues' indices
                      n_oda: int  # Number of ODA in the system
                      ) -> np.ndarray:
        """
        Allocate arrays for saving all the info.

        Parameters:
        - sol_residues: Residues in solution.

        Returns:
        - Initialized array.
            Columns are as follow:
            each atom has xyz, the center of mass also has xyx, and one
            for labeling the name of the residues, for example SOL will be 1

        Since there is chance the maximum index would be bigger than
        the number of the residues, the size of the array will be set
        to max index.
        number of row will be"
        number of frames + 1
        The extra row is for the type of the residue
        number of the columns:
        n_residues: number of the residues in solution, without residues in NP
        n_ODA: number oda residues
        NP_com: Center of mass of the nanoparticle
        than:
        timeframe + NP_com + max_residues:  xyz + n_oda * xyz
             1    +   3    +  max_residues * 3  +  n_oda * 3

        """
        rows: int = n_frames + 1  # Number of rows, 1 for name and index of res
        columns: int = 1 + 3 + max_residues * 3 + n_oda * 3
        return np.zeros((rows, columns))

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


def cleanup_mpi() -> None:
    """
    To register the cleanup function.
    """
    MPI.Finalize()


if __name__ == '__main__':

    # Register the cleanup_mpi function to be called on script exit
    atexit.register(cleanup_mpi)

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

    LOG: typing.Union[logger.logging.Logger, None]
    if RANK == 0:
        LOG = logger.setup_logger('get_frames_mpi_log')
    else:
        LOG = None
    CalculateCom(fname=sys.argv[1], log=LOG)
    CUR_TIME = datetime.datetime.now()
    FORMATED_TIME = CUR_TIME.strftime("%Y-%m-%d %H:%M:%S")
    LOG.info(FORMATED_TIME)
    print(f'{FORMATED_TIME}\n')

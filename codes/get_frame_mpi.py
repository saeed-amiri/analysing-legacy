#!/usr/bin/env python3
"""
Trajectory Analysis with MPI Parallelization

This script calculates the center of mass (COM) of residues in a mol-
ecular dynamics (MD) trajectory. It utilizes MPI parallelization to
distribute the computation across multiple processes, making it effic-
ient for large trajectory files.

The script consists of two main classes:
    1. GetResidues: Responsible for reading and processing the trajec-
       tory file, setting indices for residues based on their types,
       and storing the data for further calculation.
    2. CalculateCom: Performs the center of mass calculation for each
       frame in the trajectory using MPI parallelization.

The main steps in the script are as follows:
    1. Read the trajectory file and topology information using MDAnal-
       ysis.
    2. Preprocess the data to obtain residues in the solution and the
       nanoparticle (NP).
    3. Set indices for each residue based on its type.
    4. Utilize MPI to parallelize the computation and distribute the
       data among processes.
    5. Calculate the center of mass for each frame in the trajectory
       for different types of residues.
    6. Gather the results and combine them into a final array.

Requirements:
    - MDAnalysis: Required to read the trajectory file and topology.
    - mpi4py: Required for MPI parallelization.
    - numpy: Required for data manipulation.

Usage:
    python script_name.py trajectory_file.xtc

Note:
    Make sure you have the required dependencies installed before run-
    ning the script.

Main Classes and Methods:
-------------------------

Class GetResidues:
------------------

    Methods:
    - __init__(self, fname: str, log: Union[Logger, None]) -> None:
        Initializes the GetResidues class by reading the trajectory
        file and topology.
        Parameters:
            - fname (str): Name of the trajectory file.
            - log (Union[Logger, None]): Logger object for logging
              messages.

    - _initiate_reading(self, fname: str, log: Union[Logger, None]
                        ) -> None:
        Reads the trajectory file and topology using MDAnalysis.
        Parameters:
            - fname (str): Name of the trajectory file.
            - log (Union[Logger, None]): Logger object for logging
              messages.

    - _initiate_data(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        Processes the data to obtain residues in the solution and NP.
        Returns:
            Tuple containing dictionaries with residue indices as keys
            and residue types as values.

    - set_residues_index(self, all_res_tmp: Dict[str, List[int]]
                         ) -> Dict[int, int]:
        Sets indices for each residue based on its type.
        Parameters:
            - all_res_tmp (Dict[str, List[int]]): Dictionary with
              residue types as keys and residue indices as values.
        Returns:
            Dictionary containing residue indices as keys and their
            corresponding residue types as values.

    - get_residues(self, res_name: List[str]
                   ) -> Tuple[Dict[str, List[int]], int, int, int]:
        Returns a dictionary of residues in the solution, excluding NP
        residues.
        Parameters:
            - res_name (List[str]): List of residue names to consider.
        Returns:
            Tuple containing a dictionary of residues, the number of
            residues, and the maximum and minimum indices.

Class CalculateCom:
-------------------

    Methods:
    - __init__(self, fname: str, log: Union[Logger, None]) -> None:
        Initializes the CalculateCom class and initiates data and
        calculations.
        Parameters:
            - fname (str): Name of the trajectory file.
            - log (Union[Logger, None]): Logger object for logging
              messages.

    - _initiate_data(self, fname: str, log: Union[Logger, None]
                     ) -> None:
        Calls the GetResidues class and gets the required data from it.
        Parameters:
            - fname (str): Name of the trajectory file.
            - log (Union[Logger, None]): Logger object for logging
              messages.

    - _initiate_calc(self) -> None:
        Divides the list of timesteps and broadcasts the data between
        processes.
        Computes the center of mass for each frame in the trajectory
        using MPI parallelization.

    - process_trj(self, chunk_tstep, u_traj, np_res_ind, my_data,
                  sol_residues, amino_odn_index
                  ) -> Union[np.ndarray, None]:
        Processes the trajectory data and calculates the center of
        mass
        for each frame.
        Parameters:
            - chunk_tstep: List of timesteps assigned to the current
            process.
            - u_traj: MDAnalysis trajectory object.
            - np_res_ind: List of residue indices in the nanoparticle.
            - my_data: NumPy array to store the center of mass data.
            - sol_residues: Dictionary of solution residues.
            - amino_odn_index: Dictionary of indices for ODN amino
              group center of mass.
        Returns:
            NumPy array containing the center of mass data for the
            assigned timesteps.

    - get_com_all(self, all_atoms, ind) -> Union[np.ndarray, None]:
        Calculates the center of mass of all residues in a given frame.
        Parameters:
            - all_atoms: NumPy array containing positions of all atoms
              in the frame.
            - ind: Index of the residue.
        Returns:
            NumPy array containing the center of mass for all residues.

    - get_odn_amino_com(self, all_atoms, ind
                        ) -> Union[np.ndarray, None]:
        Calculates the center of mass of the amino group in ODN
        residues.
        Parameters:
            - all_atoms: NumPy array containing positions of all atoms
              in the frame.
            - ind: Index of the residue.
        Returns:
            NumPy array containing the center of mass for the ODN
            amino group.

    - get_np_com(self, all_atoms, np_res_ind, u_traj
                 ) -> Union[np.ndarray, None]:
        Calculates the center of mass for the nanoparticle residues in
        a given frame.
    Parameters:
        - all_atoms: NumPy array containing positions of all atoms in
          the frame.
        - np_res_ind: List of residue indices in the nanoparticle.
        - u_traj: MDAnalysis trajectory object.
    Returns:
        NumPy array containing the center of mass for the nanoparticle
        residues.

Utility Functions:
------------------

    - broadcaste_arg(self, val: Any) -> Any:
        MPI broadcast function to broadcast data from the root process
        to all other processes.
        Parameters:
            - val (Any): Data to be broadcasted.
        Returns:
            Broadcasted data.

    - set_amino_odn_index(self, all_res_tmp) -> Dict[int, int]:
        Sets indices for ODN amino groups.
        Parameters:
            - all_res_tmp (Dict[str, List[int]]): Dictionary with
            residue types as keys and residue indices as values.
        Returns:
            Dictionary containing residue indices as keys and their
            corresponding residue types as values.

    - mk_allocation(self, size: int, rank: int, num_procs: int
                   ) -> List[int]:
        Computes the data distribution among processes for MPI
        parallelization.
        Parameters:
            - size (int): Total data size.
            - rank (int): Rank of the current process.
            - num_procs (int): Total number of processes.
        Returns:
        List of data indices assigned to the current process.
    Jul 21 2023
"""


import sys
import typing
import datetime
import atexit
import pickle
from mpi4py import MPI
import numpy as np
import logger
import static_info as stinfo
import get_topo as topo
import my_tools
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
        """
        Initialize GetResidues class.

        Args:
            fname (str): Name of the trajectory file.
            log (Union[Logger, None]): Logger object for logging messages.
        """
        self._initiate_reading(fname, log)
        self.sol_res, self. np_res = self._initiate_data(log)
        self.__write_msg(log)

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: typing.Union[logger.logging.Logger, None]
                          ) -> None:
        """
        Call the other modules and read the files
        Top contains info from forcefield parameters files, the numbers
        of the residues.
        ttr_info contains trajectory read by MDAnalysis.
        Args:
            fname (str): Name of the trajectory file.
            log (Union[Logger, None]): Logger object for logging messages.
        """
        if log is not None:
            self.top = topo.ReadTop(log)
            self.trr_info = GetInfo(fname, log=log)

    def _initiate_data(self,
                       log: logger.logging.Logger
                       ) -> tuple[dict[int, int], dict[int, int]]:
        """
        Initialize data to obtain the center of mass of residues.
        MDAnalysis fails to manage indices, so NP (COR & APT) and
        solution indices are saved and set separately.

        Algorithm:
            Retrieve all residues for nanoparticles and solution, then
            set an index for each based on type defined in static_info.

        Returns:
            Tuple of dictionaries for each set with residue index as
            keys and the residues' type as values.

        """
        sol_res_tmp: dict[str, list[int]]
        np_res_tmp: dict[str, list[int]]
        sol_res_tmp, self.nr_sol_res, self.max_res, self.min_res = \
            self.get_residues(stinfo.np_info["solution_residues"], log)
        np_res_tmp, self.nr_np_res, _, _ = \
            self.get_residues(stinfo.np_info["np_residues"], log)
        sol_res = self.set_residues_index(sol_res_tmp)
        np_res = self.set_residues_index(np_res_tmp)
        return sol_res, np_res

    @staticmethod
    def set_residues_index(all_res_tmp: dict[str, list[int]]  # Name&index
                           ) -> dict[int, int]:
        """
        Set the type of the each residue as an index and number of
        residues
        """
        return \
            {res: stinfo.reidues_id[k] for k, val in all_res_tmp.items()
             for res in val}

    def get_residues(self,
                     res_name: list[str],  # Name of the residues
                     log: logger.logging.Logger
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
        if self.check_similar_items(all_res_dict):
            msg: str = (f'{self.__class__.__name__}:\n'
                        '\tError: There is a douplicate index in the '
                        'residues. Most modifying the code!\n')
            log.error(msg)
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        nr_residues = int(sum(len(lst) for lst in all_res_dict.values()))
        max_res = int(max(max(lst) for lst in all_res_dict.values()))
        min_res = int(min(min(lst) for lst in all_res_dict.values()))
        self.info_msg += \
            (f'\tThe number of the read residues for {res_name} is:\n'
             f'\t\t`{nr_residues}`\n'
             f'\t\tThe max & min of indices are: `{max_res}` and'
             f' `{min_res}`\n')
        return all_res_dict, nr_residues, max_res, min_res

    @staticmethod
    def check_similar_items(dic: dict[str, list[int]]) -> bool:
        """
        Create an empty set to store all unique elements
        """
        unique_elements: set[typing.Any] = set()

        # Iterate over the lists in the dictionary values
        for lst in dic.values():
            # Convert the list to a set to remove duplicates
            unique_set = set(lst)

            # Check if there are any common elements with the previous sets
            if unique_set & unique_elements:
                return True
            # Update the set of unique elements
            unique_elements |= unique_set

        # If no common elements were found, return False
        return False

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

    Attributes:
        info_msg (str): Messages from CalculateCom for logging.
        get_residues (GetResidues or None): Objects of the GetResidues
        class.

    Args:
        fname (str): Name of the trajectory files.
        log (logging.Logger or None): Logger for logging messages
        (optional).

    Note:
        The class calculates the center of mass for each frame in the
        trajectory
        and stores the data in a 2D array with columns representing
        time,
        center of mass of the nanoparticle, and center of mass for each
        residue.

        The array layout is as follows:
        | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ... |
         resN_x | resN_y | resN_z | odn1_x| odn1_y| odn1_z| ... odnN_z|

    Example:
        CalculateCom(fname="trajectory.trr", log=my_logger)
    """

    info_msg: str = 'Messages from CalculateCom:\n'  # To log
    get_residues: typing.Union[GetResidues, None]  # Type of the info
    n_frames: int  # Number of the frame in the trajectory

    def __init__(self,
                 fname: str,  # Name of the trajectory files
                 log: typing.Union[logger.logging.Logger, None]
                 ) -> None:
        """
        Initialize CalculateCom and perform calculations.

        Args:
            fname (str): Name of the trajectory files.
            log (logging.Logger or None, optional): Logger for logging
            messages.
        """
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
                self.get_solution_residues(stinfo.np_info['solution_residues'])
            residues_index_dict: typing.Union[dict[int, int], None] = \
                self.mk_residues_dict(sol_residues)
            if self.get_residues is not None:
                u_traj = self.get_residues.trr_info.u_traj
                com_arr: np.ndarray = \
                    self.mk_allocation(self.n_frames,
                                       self.get_residues.nr_sol_res,
                                       self.get_residues.top.mols_num['ODN'])
                if com_arr is not None and sol_residues is not None:
                    _, com_col = np.shape(com_arr)
                    amino_odn_index: typing.Union[dict[int, int], None] = \
                        self.set_amino_odn_index(com_arr, sol_residues['ODN'])

        else:
            residues_index_dict = None
            amino_odn_index = None
            sol_residues = None
            chunk_tstep = None
            np_res_ind = None
            com_col = None
            u_traj = None

        # Setting the type
        chunk_tstep = typing.cast(typing.List[typing.Any], chunk_tstep)
        # Broadcast and scatter all the data
        chunk_tstep = COMM.scatter(chunk_tstep, root=0)

        np_res_ind, com_col, u_traj, sol_residues, amino_odn_index, \
            residues_index_dict = \
            self.broadcaste_arg(np_res_ind,
                                com_col,
                                u_traj,
                                sol_residues,
                                amino_odn_index,
                                residues_index_dict)

        if chunk_tstep is not None:
            chunk_size = len(chunk_tstep)

        my_data = np.empty((chunk_size, com_col)) if \
            chunk_tstep is not None else None

        my_data = self.process_trj(
                                   chunk_tstep[-1:],
                                   u_traj,
                                   np_res_ind,
                                   my_data,
                                   sol_residues,
                                   amino_odn_index,
                                   residues_index_dict
                                   )
        COMM.barrier()
        # Gather my_data from all processes into my_data_list
        my_data_list = COMM.gather(my_data, root=0)
        # On the root process, concatenate all arrays in my_data_list
        if RANK == 0:
            recvdata: np.ndarray = np.concatenate(my_data_list, axis=0)
            com_arr = \
                self.set_residue_ind(com_arr, recvdata, residues_index_dict)
            com_arr = self.set_residue_type(com_arr, sol_residues)
            self.pickle_arr(com_arr, log=LOG)
        # Set the info_msg
        self.get_processes_info(RANK, chunk_tstep)

    def pickle_arr(self,
                   com_arr: np.ndarray,  # Array of the center of mass
                   log: logger.logging.Logger  # Name of the log file
                   ) -> None:
        """
        check the if the previus similar file exsitance the pickle
        data into a file
        """
        fname: str  # Name of the file to pickle to
        fname = my_tools.check_file_reanme(stinfo.files['com_pickle'], log)
        with open(fname, 'wb') as f_arr:
            pickle.dump(com_arr, f_arr)

    def set_residue_type(self,
                         com_arr: np.ndarray,  # Updated array to set the type
                         sol_residues: typing.Union[dict[str, list[int]], None]
                         ) -> np.ndarray:
        """
        I need to assign types to all residues and place them in the
        final row of the array.
        Args:
            com_arr: Filled array with information about COM and real
                     index
            sol_residues: key: Name of the residue
                          Value: Residues belongs to the Key
        Return:
            Updated com_arr with type of each residue in the row below
            them.
        """
        if sol_residues is not None:
            reverse_mapping = {}
            for key, value_list in sol_residues.items():
                for num in value_list:
                    reverse_mapping[num] = key
            for ind in range(com_arr.shape[1]):
                try:
                    res_ind = int(com_arr[-2, ind])
                    res_name: str = reverse_mapping.get(res_ind)
                    com_arr[-1, ind] = stinfo.reidues_id[res_name]
                except KeyError:
                    pass
            return com_arr
        return None


    @staticmethod
    def set_residue_ind(com_arr: np.ndarray,  # The final array
                        recvdata: np.ndarray,  # Info about time frames
                        residues_index_dict: typing.Union[dict[int, int], None]
                        ) -> typing.Union[np.ndarray, None]:
        """
        Set the original residues' indices to the com_arr[-2]
        Set the type of residues' indices to the com_arr[-1]
        """
        # Copy data to the final array
        for row in recvdata:
            tstep = int(row[0])
            com_arr[tstep] = row.copy()
        if residues_index_dict is not None:
            # setting the index of NP and ODA Amino heads
            com_arr[-2, 1:4] = [-1, -1, -1]
            com_arr[-2, -50:] = np.arange(-1, -51, -1)
            for res_ind, col_in_arr in residues_index_dict.items():
                ind = int(res_ind)
                com_arr[-2][col_in_arr:col_in_arr+3] = \
                    np.array([ind, ind, ind]).copy()
            return com_arr
        return None

    def process_trj(self,
                    chunk_tstep,  # Frames' ind
                    u_traj,  # Trajectory
                    np_res_ind: typing.Union[list[int], None],  # NP residue id
                    my_data: typing.Union[np.ndarray, None],  # To save COMs
                    sol_residues: typing.Union[dict[str, list[int]], None],
                    amino_odn_index: typing.Union[dict[int, int], None],
                    residues_index_dict: typing.Union[dict[int, int], None]
                    ) -> typing.Union[np.ndarray, None]:
        """Get atoms in the timestep"""
        if (
           chunk_tstep is not None and
           my_data is not None and
           sol_residues is not None and
           residues_index_dict is not None
           ):
            for row, i in enumerate(chunk_tstep):
                ind = int(i)
                frame = u_traj.trajectory[ind]
                atoms_position: np.ndarray = frame.positions
                for k, val in sol_residues.items():
                    for item in val:
                        com = self.get_com_all(atoms_position, item)
                        if com is None:
                            continue  # Skip if com is None

                        wrap_com = self.wrap_position(com, frame.dimensions)

                        element = residues_index_dict[item]
                        my_data[row][element:element+3] = wrap_com

                        if k == 'ODN':
                            if amino_odn_index is not None:
                                amin = self.get_odn_amino_com(atoms_position,
                                                              item)
                                amino_ind: int = amino_odn_index[item]
                                my_data[row][amino_ind:amino_ind+3] = amin
                np_com = self.get_np_com(atoms_position, np_res_ind, u_traj)
                # Update my_data with ind and np_com values
                my_data[row, 0] = ind
                my_data[row, 1:4] = np_com
            return my_data
        return None

    def get_com_all(self,
                    all_atoms: np.ndarray,  # All the atoms position
                    ind: int  # Index of the residue
                    ) -> typing.Union[np.ndarray, None]:
        """
        return all the residues' center of mass

        Algorithem:
            Atoms belong to the residue `ind` is save to:
                i_residue: <class 'MDAnalysis.core.groups.AtomGroup'>
                e.g.:
                <AtomGroup [<Atom 1: OH2 of type O of resname SOL,
                             resid 1803 and segid SYSTEM>,
                            <Atom 2: H1 of type H of resname SOL,
                             resid 1803 and segid SYSTEM>,
                            <Atom 3: H2 of type H of resname SOL,
                             resid 1803 and segid SYSTEM>]>

            Indicis of these atoms are:
                atom_indices: <class 'numpy.ndarray'>
                e.g.: [0 1 2]

            Positions of all the atoms in the residues are:
                atom_positions: <class 'numpy.ndarray'>
                e.g.:
                     [[186.44977   20.240488  14.459618]
                     [185.91147   20.554382  13.733033]
                     [186.8721    21.027851  14.803037]]

            Masses of these atoms are also found:
                atom_masses: <class 'numpy.ndarray'>
                e.g.: [15.999  1.008  1.008]
        Return:
            By getting weightesd average of these atoms the center of
            mass is returend
        """
        if self.get_residues is not None:
            i_residue = \
                self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
            atom_indices = i_residue.indices
            atom_positions = all_atoms[atom_indices]
            atom_masses = i_residue.masses
            return np.average(atom_positions, weights=atom_masses, axis=0)
        return None

    def get_odn_amino_com(self,
                          all_atoms: np.ndarray,  # All the atoms poistion
                          ind: int  # Index of the residue
                          ) -> typing.Union[np.ndarray, None]:
        """
        Calculating the center of mass of the amino group in the ODN
        See the doc of get_com_all
        Here, only atoms in NH2 and HC (3 H atoms)
        i_atoms:
            <Atom 523033: NH2 of type N of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523034: HC1 of type H of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523035: HC2 of type H of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523036: HC3 of type H of resname ODN, resid 176274
             and segid SYSTEM>
        """
        if self.get_residues is not None:
            i_residue = \
                self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
            i_atoms = i_residue.select_atoms('name NH2 HC*')
            atom_indices = i_atoms.indices
            atom_positions = all_atoms[atom_indices]
            atom_masses = i_atoms.masses
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
            if info_msg_all is not None:
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

    def get_solution_residues(self,
                              res_group: list[str]
                              ) -> typing.Union[dict[str, list[int]], None]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        sol_dict: dict[str, list[int]] = {}  # All the residues in solution
        if self.get_residues is not None:
            for k, val in self.get_residues.trr_info.residues_indx.items():
                if k in res_group:
                    sol_dict[k] = val
            return sol_dict
        return None

    @staticmethod
    def mk_residues_dict(sol_residues: typing.Union[dict[str, list[int]], None]
                         ) -> typing.Union[dict[int, int], None]:
        """
        Make a dict for indexing all the residues. Not always residues
        indexed from zero and/or are numberd sequently.

        Args:
            sol_residues of index for each residue in the solution
        Return:
            new indexing for each residues
            Since in the recived method of this return, the result could
            be None, the type is Union
            Key: The residue index in the main data (traj from MDAnalysis)
            Value: The new orderd indices
        Notes:
            Since we already have 4 elements before the these resideus,
            numbering will start from 4
        """
        if sol_residues is not None:
            all_residues: list[int] = \
                [item for sublist in sol_residues.values() for item in sublist]
            sorted_residues: list[int] = sorted(all_residues)
            residues_index_dict: typing.Union[dict[int, int], None] = {}
            if residues_index_dict is not None:
                for i, res in enumerate(sorted_residues):
                    residues_index_dict[res] = i * 3 + 4
            return residues_index_dict
        return None

    @staticmethod
    def set_amino_odn_index(com_arr,  # The array to set all com in it
                            odn_residues: list[int]  # Indices of ODN residues
                            ) -> dict[int, int]:
        """
        Set (or find!) the indices for the COM of ODN amino group in
        the com_arr
        In the alocation of com_arr, nr_odn columns are added for the
        com of the amino groups of the ODN. Since the COM of all ODN
        are also set in the com_arr the extra indices should carefuly
        be setted.
        The indices of the ODN could be not sequal!
        Returns:
            key: 0 - nr_odn
            value: the column in the com_arr (the final arry)
        """
        sorted_odn_residues: list[int] = sorted(odn_residues, reverse=True)
        last_column: int = np.shape(com_arr)[1]
        odn_amino_indices: dict[int, int] = {}
        for i, odn in enumerate(sorted_odn_residues):
            odn_amino_indices[odn] = int(last_column - (i+1) * 3)
        return odn_amino_indices

    @staticmethod
    def broadcaste_arg(*args  # All the things that should be broadcasted
                       ) -> tuple[typing.Any, ...]:
        """
        Broadcasting data
        """
        broad_list: list[typing.Any] = []
        for arg in args:
            broad_list.append(COMM.bcast(arg, root=0))
        return tuple(broad_list)

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
                      nr_residues: int,  # Numbers of residues' indices
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

        The indexing method is updated, now every index getting a defiend
        index which is started from 4. See: mk_residues_dict
        number of row will be"
        number of frames + 1
        The extra row is for the type of the residue
        number of the columns:
        n_residues: number of the residues in solution, without residues in NP
        n_ODA: number oda residues
        NP_com: Center of mass of the nanoparticle
        than:
        timeframe + NP_com + nr_residues:  xyz + n_oda * xyz
             1    +   3    +  nr_residues * 3  +  n_oda * 3

        """

        rows: int = n_frames + 2  # Number of rows, 2 for name and index of res
        columns: int = 1 + 3 + nr_residues * 3 + n_oda * 3
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
    current_time = datetime.datetime.now()
    formated_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    msg: str = 'Message from cleanup_mpi:\n'
    msg += f'\tFinalized at {formated_time}\n'
    print(msg)
    if LOG is not None:
        LOG.info(msg)


if __name__ == '__main__':

    # Register the cleanup_mpi function to be called on script exit
    LOG: typing.Union[logger.logging.Logger, None]
    atexit.register(cleanup_mpi)

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

    if RANK == 0:
        LOG = logger.setup_logger('get_frames_mpi_log')
    else:
        LOG = None
    CalculateCom(fname=sys.argv[1], log=LOG)

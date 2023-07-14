"""this script uses the data from get_trajectory.py and returns the
residues' positions in the 3d matrix. Then it will; it will get the
center of mass of each residue at each timestep.
The scripts initiated by chatGpt

It reads the unwraped trr files
Currently, the output is a binary file in numpy's pickled format. The
array contains rows equivalent to the number of time frames, with each
row representing a time step in the tar file divided by the frequency
of the saving trajectory (set in static_info.py). I have observed that
the residues may have missing indexes, causing the array to have fewer
rows than expected. I have identified the largest number of indexes in
the residues to address this.
Additionally, the last row saves the index for labeling the residues
type, set in stinfo.
The array has columns equal to three times the total number of residues
plus the center of mass of the NP at each time (to save x, y, and z).
The center of the mass of all residue is calculated, then it is wrapped
back into the box, and the center of mass of the NP at that time is
subtracted from it.
02.06.2023
----------
"""

import os
import sys
import multiprocessing
import concurrent.futures
import pickle
import numpy as np
import logger
import MDAnalysis as mda
import static_info as stinfo
import my_tools
import get_topo as topo
from get_trajectory import GetInfo


class ResiduePositions:
    """getting posotions of the com the residues"""

    info_msg: str = 'Message:\n'  # To log info

    def __init__(self,
                 log: logger.logging.Logger  # Name of the log file
                 ):
        self.parra_sty = 'serial'
        self.top = topo.ReadTop(log)
        self.trr_info = GetInfo(sys.argv[1], log=log)
        self.get_center_of_mass(log)

    def get_center_of_mass(self,
                           log: logger.logging.Logger  # Name of the log file
                           ) -> None:
        """
        calculate the center mass of the each residue
        """
        # update the residues index to get the NP: APT_COR
        sol_residues: dict[str, list[int]] = self.get_solution_residues()
        com_arr: np.ndarray = self.mk_allocation(sol_residues)
        self.pickle_arr(com_arr, sol_residues, log)

    def pickle_arr(self,
                   com_arr: np.ndarray,  # Array of the center of mass
                   sol_residues: dict[str, list[int]],  # Residues in solution
                   log: logger.logging.Logger  # Name of the log file
                   ) -> None:
        """
        check the if the previus similar file exsitance the pickle
        data into a file
        """
        fname: str  # Name of the file to pickle to
        fname = my_tools.check_file_reanme(stinfo.files['com_pickle'], log)
        if self.parra_sty == 'serial':
            _com_arr = self.get_coms(com_arr, sol_residues)
        elif self.parra_sty == 'concurrent':
            _com_arr = self.get_coms_concurrent(com_arr, sol_residues)
        elif self.parra_sty == 'split':
            _com_arr = self.get_coms_parallel(com_arr, sol_residues)
        else:
            _com_arr = self.get_coms_multiprocessing(com_arr, sol_residues)
        # _com_arr = self.get_coms(com_arr, sol_residues)
        with open(fname, 'wb') as f_arr:
            pickle.dump(_com_arr, f_arr)

    # def process_tstep(self, tstep, np_res_ind, sol_residues, com_arr):
    def process_tstep(self,
                      args: tuple  # All the arguments
                      ) -> np.ndarray:
        """
        Get each timestep and do the calculations

        Parameters:
        args:
           tstep: int -> Time step of the frame
           np_res_ind:  -> Indicies of the residues
           sol_residues: dict[str, list[int]] -> Residues in solution pahse
           com_arr: np.ndarray -> Array for the center of the mass of residues
        """
        tstep, np_res_ind, sol_residues, com_arr = args
        print(tstep)
        i_step = int(tstep.time / stinfo.times['time_step'])
        all_atoms = tstep.positions
        ts_np_com = self.get_np_com(all_atoms, np_res_ind)
        com_arr[i_step][0:3] = ts_np_com
        for k, val in sol_residues.items():
            for item in val:
                com = self.get_com_all(all_atoms, item)
                wrap_com = self.wrap_position(com, tstep.dimensions)
                i_com = wrap_com - ts_np_com
                element = int(item * 3)
                com_arr[i_step][element:element + 3] = i_com
                r_idx = stinfo.reidues_id[k]
                com_arr[-1][element:element + 3] = \
                    np.array([[r_idx, r_idx, r_idx]])
        return ts_np_com

    def get_coms_concurrent(self,
                            com_arr: np.ndarray,
                            sol_residues: dict[str, list[int]]
                            ) -> np.ndarray:
        """
        Getting the COM with concurrent.futures
        """
        np_res_ind = self.get_np_residues()
        all_t_np_coms = []
        num_worker: int = 4
        self.info_msg += '\tUsing concurrent.futures module:\n'
        self.info_msg += f'\tNumber of processes is: `{os.cpu_count()}`\n'
        self.info_msg += f'\tNumber of worker is set to: `{num_worker}`\n'
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(
                                                    max_workers=num_worker
                                                    ) as executor:
            # Prepare the arguments for process_tstep function
            args_list = \
                [(tstep, np_res_ind, sol_residues, com_arr)
                 for tstep in self.trr_info.u_traj.trajectory]

            # Submit the tasks to the executor
            results = executor.map(self.process_tstep, args_list)

            # Iterate through the results and collect the ts_np_com values
            for ts_np_com in results:
                all_t_np_coms.append(ts_np_com)

        return com_arr

    def get_coms_multiprocessing(self,
                                 com_arr: np.ndarray,  # Array of the COM
                                 sol_residues: dict[str, list[int]]  # Reses
                                 ) -> np.ndarray:
        """
        Getting the COM with multiprocessing
        """
        np_res_ind = self.get_np_residues()
        all_t_np_coms = []
        num_processes: int = multiprocessing.cpu_count()
        num_worker: int = 4
        self.info_msg += '\tUsing multiprocessing module:\n'
        self.info_msg += f'\tNumber of processes is: `{num_processes}`\n'
        self.info_msg += f'\tNumber of worker is set to: `{num_worker}`\n'
        with multiprocessing.Pool(processes=num_worker) as pool:
            results = []
            for tstep in self.trr_info.u_traj.trajectory:
                args = (tstep, np_res_ind, sol_residues, com_arr)
                results.append(
                    pool.apply_async(
                                     self.process_tstep,
                                     args=(args,)
                                    )
                              )

            for result in results:
                ts_np_com = result.get()
                all_t_np_coms.append(ts_np_com)

        return com_arr

    def get_coms(self,
                 com_arr: np.ndarray,  # Zero array to save the coms
                 sol_residues: dict[str, list[int]]  # Residues in solution
                 ) -> np.ndarray:
        """
        Set the center of the mass value of each residue with its
        index in the main data and an integer as an id representing
        its residue name data. These ids are set in "stinfo.py"
        """
        np_res_ind: list[int] = []  # All the index in the NP
        all_t_np_coms: list[np.ndarray] = []  # COMs at each timestep
        np_res_ind = self.get_np_residues()
        for tstep in self.trr_info.u_traj.trajectory:
            i_step = int(tstep.time/self.trr_info.num_dict['dt'])  # time step
            all_atoms: np.ndarray = tstep.positions
            ts_np_com = self.get_np_com(all_atoms, np_res_ind)
            com_arr[i_step][0:3] = ts_np_com
            for k, val in sol_residues.items():
                for item in val:
                    com = self.get_com_all(all_atoms, item)
                    wrap_com = self.wrap_position(com, tstep.dimensions)
                    i_com = wrap_com - ts_np_com
                    element = int(item*3)
                    com_arr[i_step][element:element+3] = i_com
                    r_idx = stinfo.reidues_id[k]  # Residue idx
                    com_arr[-1][element:element+3] = \
                        np.array([[r_idx, r_idx, r_idx]])
            all_t_np_coms.append(ts_np_com)
        return com_arr

    def get_coms_parallel(self,
                          com_arr: np.ndarray,  # Array of the COM
                          sol_residues: dict[str, list[int]]  # Reses
                          ) -> np.ndarray:
        """
        Getting the COM with multiprocessing by splititng the frames
        """
        all_steps = {}
        for tstep in self.trr_info.u_traj.trajectory:
            all_steps[tstep.frame] = np.copy(tstep.positions)


    @staticmethod
    def wrap_position(pos: np.ndarray,  # The center of mass
                      vec: np.ndarray  # Box vectors
                      ) -> np.ndarray:
        """
        Wraped the position to the box for unwraped trr.
        """
        for i in range(3):
            pos[i] -= np.floor(pos[i]/vec[i])*vec[i]
        return pos

    def get_solution_residues(self) -> dict[str, list[int]]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        sol_dict: dict[str, list[int]] = {}  # All the residues in solution
        for k, val in self.trr_info.residues_indx.items():
            if k in stinfo.np_info['solution_residues']:
                sol_dict[k] = val
        return sol_dict

    def get_com_all(self,
                    all_atoms: np.ndarray,  # All the atoms position
                    ind: int  # Index of the residue
                    ) -> np.ndarray:
        """
        return all the residues com
        """
        i_residue = self.trr_info.u_traj.select_atoms(f'resnum {ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        return np.average(atom_positions, weights=atom_masses, axis=0)

    def get_np_residues(self) -> list[int]:
        """
        return list of the integer of the residues in the NP
        """
        np_res_ind: list[int] = []  # All the index in the NP
        for item in stinfo.np_info['np_residues']:
            np_res_ind.extend(self.trr_info.residues_indx[item])
        return np_res_ind

    def get_np_com(self,
                   all_atoms: np.ndarray,  # Atoms positions
                   np_res_ind: list[int]  # Index of the residues in NP
                   ) -> np.ndarray:
        """get the COM for each time step"""
        i_com: list[np.ndarray] = []  # Arrays contains center of masses
        total_mass: float = 0  # Total mass of each residue in the NP
        for i in np_res_ind:
            com: np.ndarray  # Conter of mass of the residue i
            tmp_mass: float  # Mass of the residue
            com, tmp_mass = self.get_np_com_tstep(i, all_atoms)
            total_mass += tmp_mass
            i_com.append(com)
            step_com = np.vstack(i_com)
        return np.sum(step_com, axis=0) / total_mass

    def get_np_com_tstep(self,
                         res_ind: int,  # index of the residue
                         all_atoms: np.ndarray  # Atoms positions
                         ) -> tuple[np.ndarray, float]:
        """
        calculate the center of mass of each time step for NP
        """
        i_residue: mda.core.groups.AtomGroup  # Atoms info in res
        i_residue = self.trr_info.u_traj.select_atoms(f'resnum {res_ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        tmp_mass = np.sum(atom_masses)
        com = np.average(atom_positions, weights=atom_masses,
                         axis=0) * tmp_mass
        return com, tmp_mass

    def mk_allocation(self,
                      sol_residues: dict[str, list[int]]  # residues at water
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
        """
        frames: int = self.trr_info.num_dict['n_frames']
        rows: int = frames + 1  # Number of rows, 2 for name and index of res
        max_residue = max(item for sublist in sol_residues.values() for
                          item in sublist)
        columns: int = 3 * (max_residue + 1)
        return np.zeros((rows, columns))


if __name__ == '__main__':
    positions = ResiduePositions(log=logger.setup_logger('get_frames_log'))

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
The array has columns equal to three times the total number of residues
plus the center of mass of the NP at each time (to save x, y, and z).
Additionally, the last column saves the index for labeling the residues
type, set in stinfo.
The center of the mass of all residue is calculated, then it is wrapped
back into the box, and the center of mass of the NP at that time is
subtracted from it.
02.06.2023
----------
"""

import sys
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
    def __init__(self,
                 trr_info: GetInfo,  # All the info from trr and gro files
                 log: logger.logging.Logger  # Name of the log file
                 ):
        self.info: GetInfo = trr_info
        self.top = topo.ReadTop()
        self.get_center_of_mass(log)

    def get_center_of_mass(self,
                           log: logger.logging.Logger  # Name of the log file
                           ) -> None:
        """calculate the center mass of the each residue"""
        # update the residues index to get the NP: APT_COR
        sol_residues: dict[str, list[int]] = self.__solution_residues()
        com_arr: np.ndarray = self.__allocate(sol_residues)
        self.pickle_arr(com_arr, sol_residues, log)

    def pickle_arr(self,
                   com_arr: np.ndarray,  # Array of the center of mass
                   sol_residues: dict[str, list[int]],  # Residues in solution
                   log: logger.logging.Logger  # Name of the log file
                   ) -> None:
        """check the if the previus similar file exsitance the pickle
        data into a file"""
        fname: str  # Name of the file to pickle to
        fname = my_tools.check_file_reanme(stinfo.files['com_pickle'], log)
        with open(fname, 'wb') as f_arr:
            pickle.dump(self.__get_coms(com_arr, sol_residues), f_arr)

    def __get_coms(self,
                   com_arr: np.ndarray,  # Zero array to save the coms
                   sol_residues: dict[str, list[int]]  # Residues in solution
                   ) -> np.ndarray:
        """set the center of the mass value of each residue with its
        index in the main data and an integer as an id representing
        its residue name data. These ids are set in "stinfo.py" """
        np_res_ind: list[int] = []  # All the index in the NP
        all_t_np_coms: list[np.ndarray] = []  # COMs at each timestep
        np_res_ind = self.__np_rsidues()
        for tstep in self.info.u_traj.trajectory:
            i_step = int(tstep.time/stinfo.times['time_step'])  # time step
            all_atoms: np.ndarray = tstep.positions
            print(f'\n{tstep.time}:')
            ts_np_com = self.__np_com(all_atoms, np_res_ind)
            com_arr[i_step][0:3] = ts_np_com
            for k, val in sol_residues.items():
                for item in val:
                    com = self.__get_com_all(all_atoms, item)
                    wrap_com = self.__wrap_position(com, tstep.dimensions)
                    i_com = wrap_com - ts_np_com
                    element = int(item*3)
                    com_arr[i_step][element:element+3] = i_com
                com_arr[i_step][-1] = stinfo.reidues_id[k]
            all_t_np_coms.append(ts_np_com)
        return com_arr

    def __wrap_position(self,
                        pos: np.ndarray,  # The center of mass
                        vec: np.ndarray  # Box vectors
                        ) -> np.ndarray:
        """wraped the position to the box for unwraped trr"""
        for i in range(3):
            pos[i] -= np.floor(pos[i]/vec[i])*vec[i]
        return pos

    def __solution_residues(self) -> dict[str, list[int]]:
        """return the dict of the residues in the solution with
        dropping the NP residues"""
        sol_dict: dict[str, list[int]] = {}  # All the residues in solution
        for k, val in self.info.residues_indx.items():
            if k in stinfo.np_info['solution_residues']:
                sol_dict[k] = val
        return sol_dict

    def __get_com_all(self,
                      all_atoms: np.ndarray,  # All the atoms position
                      ind: int  # Index of the residue
                      ) -> np.ndarray:
        """return all the residues com"""
        i_residue = self.info.u_traj.select_atoms(f'resnum {ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        return np.average(atom_positions, weights=atom_masses, axis=0)

    def __np_rsidues(self) -> list[int]:
        """return list of the integer of the residues in the NP"""
        np_res_ind: list[int] = []  # All the index in the NP
        for item in stinfo.np_info['np_residues']:
            np_res_ind.extend(self.info.residues_indx[item])
        return np_res_ind

    def __np_com(self,
                 all_atoms: np.ndarray,  # Atoms positions
                 np_res_ind: list[int]  # Index of the residues in NP
                 ) -> np.ndarray:
        """get the COM for each time step"""
        i_com: list[np.ndarray] = []  # Arrays contains center of masses
        total_mass: float = 0  # Total mass of each residue in the NP
        for i in np_res_ind:
            com: np.ndarray  # Conter of mass of the residue i
            tmp_mass: float  # Mass of the residue
            com, tmp_mass = self.__get_np_com(i, all_atoms)
            total_mass += tmp_mass
            i_com.append(com)
            step_com = np.vstack(i_com)
        return np.sum(step_com, axis=0) / total_mass

    def __get_np_com(self,
                     res_ind: int,  # index of the residue
                     all_atoms: np.ndarray  # Atoms positions
                     ) -> tuple[np.ndarray, float]:
        """calculate the center of mass of each time step for NP"""
        i_residue: mda.core.groups.AtomGroup  # Atoms info in res
        i_residue = self.info.u_traj.select_atoms(f'resnum {res_ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        tmp_mass = np.sum(atom_masses)
        com = np.average(atom_positions, weights=atom_masses,
                         axis=0) * tmp_mass
        return com, tmp_mass

    def __allocate(self,
                   sol_residues: dict[str, list[int]]  # residues in solution
                   ) -> np.ndarray:
        """allocate arraies for saving all the info"""
        frames: int = self.info.num_dict['n_frames']
        rows: int = frames + 2  # Number of rows, 2 for name and index of res
        # Columns are as follow:
        # each atom has xyz, the center of mass also has xyx, and one
        # for labeling the name of the residues, for example SOL will be 1
        max_residue = max(item for sublist in sol_residues.values() for
                          item in sublist)
        columns: int = 3 * (max_residue + 1) + 1
        return np.zeros((rows, columns))


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('get_frames_log'))
    positions = ResiduePositions(trr,
                                 log=logger.setup_logger('get_frames_log'))

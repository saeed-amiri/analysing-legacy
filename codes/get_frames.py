"""this script uses the data from get_trajectory.py and returns the
residues' positions in the 3d matrix. Then it will; it will get the
center of mass of each residue at each timestep.
The scripts initiated by chatGpt
"""

import sys
import numpy as np
import logger
import MDAnalysis as mda
import static_info as stinfo
import get_topo as topo
from get_trajectory import GetInfo


class ResiduePositions:
    """getting posotions of the """
    def __init__(self,
                 trr_info: GetInfo  # All the info from trr and gro files
                 ):
        self.info: GetInfo = trr_info
        self.top = topo.ReadTop()
        self.get_center_of_mass()

    def get_center_of_mass(self) -> None:
        """calculate the center mass of the each residue"""
        # update the residues index to get the NP: APT_COR
        com_arr: np.ndarray = self.__allocate()
        self.__get_coms(com_arr)

    def __get_coms(self,
                   com_arr: np.ndarray  # Zero array to save the coms
                   ) -> np.ndarray:
        """set the center of the mass value of each residue with its
        index in the main data and an integer as an id representing
        its residue name data. These ids are set in "stinfo.py" """
        np_res_ind: list[int] = []  # All the index in the NP
        all_t_np_coms: list[np.ndarray] = []  # COMs at each timestep
        np_res_ind = self.__np_rsidues()
        for tstep in self.info.u_traj.trajectory:
            if tstep.time < 20.0:
                i_step = int(tstep.time)  # Current time step
                all_atoms: np.ndarray = tstep.positions
                print(f'\n{tstep.time}:')
                ts_np_com = self.__np_com(all_atoms, np_res_ind)
                com_arr[i_step][0:3] = ts_np_com
                for k, val in self.info.residues_indx.items():
                    for item in val:
                        com = self.__get_com_all(all_atoms, item) - ts_np_com
                        element = int((item+1)*3)
                        com_arr[i_step][element:element+3] = com
                    com_arr[i_step][-1] = stinfo.reidues_id[k]
                all_t_np_coms.append(ts_np_com)
        np_coms: np.ndarray = np.vstack(all_t_np_coms)
        return np_coms

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

    def __allocate(self) -> np.ndarray:
        """allocate arraies for saving all the info"""
        frames: int = self.info.num_dict['n_frames']
        rows: int = frames + 2  # Number of rows, 2 for name and index of res
        # Columns are as follow:
        # each atom has xyz, the center of mass also has xyx, and one
        # for labeling the name of the residues, for example SOL will be 1
        columns: int = 3 * (sum(v for v in self.top.mols_num.values()) + 1) + 1
        return np.zeros((rows, columns))


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('get_frames_log'))
    positions = ResiduePositions(trr)

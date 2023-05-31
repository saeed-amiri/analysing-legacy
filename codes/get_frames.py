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
        residues_indx: dict[str, list[int]] = self.__get_np_index()
        com_arr: np.ndarray = self.__allocate()
        self.__get_coms(com_arr)

    def __get_coms(self,
                   com_arr: np.ndarray  # Zero array to save the coms
                   ) -> np.ndarray:
        """set the center of the mass value of each residue with its
        index in the main data and an integer as an id representing
        its residue name data. These ids are set in "stinfo.py" """
        np_res_ind: list[int] = []  # All the index in the NP
        all_coms: list[np.ndarray] = []  # COMs at each timestep
        np_res_ind = self.__np_rsidues()
        for tstep in self.info.u_traj.trajectory:
            all_atoms: np.ndarray = tstep.positions
            print(f'\n{tstep.time}:\n')
            ts_np_com = self.__np_com(all_atoms, np_res_ind)
            for k, val in self.info.residues_indx.items():
                for item in val:
                    i_residue = self.info.u_traj.select_atoms(f'resnum {item}')
                    atom_indices = i_residue.indices
                    atom_positions = all_atoms[atom_indices]
                    atom_masses = i_residue.masses
                    print(k, item, np.average(atom_positions, weights=atom_masses,
                         axis=0))
            all_coms.append(ts_np_com)
        np_coms: np.ndarray = np.vstack(all_coms)
        return np_coms

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
        columns: int = sum(v for v in self.top.mols_num.values())
        return np.zeros((rows, columns))

    def __get_np_index(self) -> dict[str, list[int]]:
        residues_indx: dict[str, list[int]] = {}
        npi: str = stinfo.np_info['np_residues'][0]
        npj: str = stinfo.np_info['np_residues'][1]
        for k in stinfo.reidues_num.keys():
            if k in stinfo.np_info['solution_residues']:
                residues_indx[k] = self.info.residues_indx[k].copy()
        residues_indx[stinfo.np_info['np_name']] = \
            self.info.residues_indx[npi].copy() + \
            self.info.residues_indx[npj].copy()
        return residues_indx


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('get_frames_log'))
    positions = ResiduePositions(trr)

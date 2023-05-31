"""this script uses the data from get_trajectory.py and returns the
residues' positions in the 3d matrix. Then it will; it will get the
center of mass of each residue at each timestep.
The scripts initiated by chatGpt
"""

import sys
import numpy as np
import logger
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
        all_coms = []
        for item in stinfo.np_info['np_residues']:
            np_res_ind.extend(self.info.residues_indx[item])

        for tstep in self.info.u_traj.trajectory:
            count = 0
            x_com = []
            total_mass: float = 0
            all_atoms: np.ndarray = tstep.positions
            print(f'\n{tstep.time}:\n')
            for i in np_res_ind:
                i_residue = self.info.u_traj.select_atoms(f'resnum {i}')
                atom_indices = i_residue.indices
                atom_positions = all_atoms[atom_indices]
                atom_masses = i_residue.masses
                tmp_mass = np.sum(atom_masses)
                com = np.average(atom_positions, weights=atom_masses,
                                 axis=0) * tmp_mass
                count += 1
                total_mass += tmp_mass
                x_com.append(com)

            all_com = np.vstack(x_com)
            all_coms.append(np.sum(all_com, axis=0)/total_mass)
        coms = np.vstack(all_coms)
        return coms

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

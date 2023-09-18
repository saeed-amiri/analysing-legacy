"""
Analysing water residues
1- Finding the surface position of the water at the interface
"""

import numpy as np
import pandas as pd

import logger
import static_info as stinfo
from get_data import GetData
from colors_text import TextColor as bcolors


class WrapData(GetData):
    """
    Get data and call other classes to analysis and plot them.
    Before that, some calculateion must be done
    """

    mean_nanop_com: np.ndarray  # Average of the nanoparticle COM over times
    # Shift of COM at each time from average
    nanoparticle_disp_from_avg_com: np.ndarray
    interface_locz: float  # Z location of the interface
    nanop_radius: float  # Radius of the nanoparticle

    def __init__(self) -> None:
        super().__init__()
        self.set_constants()
        self._initiate_calc()

    def set_constants(self) -> None:
        """
        Set the constants for all the uses
        """
        self.interface_locz = 113.9
        self.nanop_radius = stinfo.np_info['radius']

    def _initiate_calc(self) -> None:
        """
        Initiate calculation and some analysis which are needed for
        all other classes
        """
        self.mean_nanop_com = self.find_mean_of_np_com()
        self.nanoparticle_disp_from_avg_com = self.find_np_shift_from_mean()

    def find_mean_of_np_com(self) -> np.ndarray:
        """find mean of the nanoparticle center of mass"""
        return np.mean(self.split_arr_dict['APT_COR'], axis=0)

    def find_np_shift_from_mean(self,
                                ) -> np.ndarray:
        """find the shift of com of nanoparticle from mean value at
        each time step"""
        return self.split_arr_dict['APT_COR'] - self.mean_nanop_com


class GetSurface:
    """
    To determine the level of APTES protonation, locating the water's
    surface is necessary. This requires inputting all nanoparticle data
    (APT_COR) for accurate placement.
    The script identifies the location of each water molecule based on
    the COM the water molecules.

    To proceed, we rely on two assumptions:
        1- The water atoms situated above the nanoparticles' highest
           point are considered to belong to the bottom of the system
           as they will eventually fall from the dataframe.
        2- Additionally, any water molecule within or around the
           nanoparticle, enclosed by the smallest sphere, should not be
           considered part of the interface and should be removed.
    ."""

    info_msg: str = 'Message from GetSurface:\n'  # Message for logging
    # Set in "get_water_surface" method:
    interface_z: np.float64  # Average place of the water suraface at interface
    interface_std: np.float64  # standard diviasion of the water suraface
    contact_angle: np.float64  # Final contact angle

    def __init__(self,
                 ) -> None:
        all_data = WrapData()
        self.get_interface(all_data)
        # self.__write_msg(log)
        # self.info_msg = ''  # Empety the msg

    def get_interface(self,
                      all_data: WrapData
                      ) -> None:
        """get the water surface"""

        residues_atoms: dict[str, pd.DataFrame]  # All atoms in ress
        residues_atoms = all_data.split_arr_dict
        water_residues: np.ndarray = residues_atoms['SOL'][:-2]
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        mesh_size: np.float64
        for frame in range(water_residues.shape[0]):
            x_mesh, y_mesh, mesh_size = \
                self._get_grid_xy(
                    water_residues[frame, ::3], water_residues[frame, 1::3])

    @staticmethod
    def _get_grid_xy(x_data: np.ndarray,  # x component of the coms
                     y_data: np.ndarray,  # y component of the coms
                     ) -> tuple[np.ndarray, np.ndarray, np.float64]:
        """return the mesh grid for the xy of sol"""
        x_min: np.float64 = np.min(x_data)
        y_min: np.float64 = np.min(y_data)
        x_max: np.float64 = np.max(x_data)
        y_max: np.float64 = np.max(y_data)
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        mesh_size: np.float64 = (x_max-x_min)/100.
        x_mesh, y_mesh = np.meshgrid(
            np.arange(x_min, x_max + mesh_size, mesh_size),
            np.arange(y_min, y_max + mesh_size, mesh_size))
        return x_mesh, y_mesh, mesh_size

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetSurface.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    GetSurface()

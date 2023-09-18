"""
To analysing the behavior of ODA at the interface
"""

import numpy as np

import logger
from get_data import GetData
import static_info as stinfo


class WrapData(GetData):
    """
    Get data and call other classes to analysis and plot them.
    Before that, some calculateion must be done
    """

    mean_nanop_com: np.ndarray  # Average of the nanoparticle COM over times
    shift_nanop_com: np.ndarray  # Shift of COM at each time from average
    interface_locz: float  # Z location of the interface
    nanop_radius: float  # Radius of the nanoparticle

    def __init__(self) -> None:
        super().__init__()
        self.set_constants()
        self.initiate_calc()

    def set_constants(self) -> None:
        """
        Set the constants for all the uses
        """
        self.interface_locz = 113.9
        self.nanop_radius = stinfo.np_info['radius']

    def initiate_calc(self) -> None:
        """
        Initiate calculation and some analysis which are needed for
        all other classes
        """
        self.mean_nanop_com = self.find_mean_of_np_com()
        self.shift_nanop_com = self.find_np_shift_from_mean()

    def find_mean_of_np_com(self) -> np.ndarray:
        """find mean of the nanoparticle center of mass"""
        return np.mean(self.split_arr_dict['APT_COR'], axis=0)

    def find_np_shift_from_mean(self,
                                ) -> np.ndarray:
        """find the shift of com of nanoparticle from mean value at
        each time step"""
        return self.split_arr_dict['APT_COR'] - self.mean_nanop_com


class OdaAnalysis(WrapData):
    """
    A class for analyzing ODN data and creating various plots.

    This class inherits from the `WrapData` class and provides methods
    to analyze ODN data and generate plots related to ODN densities,
    center of mass, and more.

    Attributes:
        None

    Methods:
        __init__(): Initialize the PlotOdnAnalysis instance.
        plot_average_annulus_density(counts: np.ndarray, radii_distance:
            np.ndarray) -> None:
            Plot the average ODN density across annuluses.
        plot_smoothed_annulus_density(counts: np.ndarray, radii_distance:
            np.ndarray) -> None:
            Plot smoothed ODN density across annuluses.
        plot_odn(odn_arr: np.ndarray) -> None:
            Plot the center of mass of ODN molecules.
    """

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__()


if __name__ == "__main__":
    OdaAnalysis(log='oda_analysis.log')

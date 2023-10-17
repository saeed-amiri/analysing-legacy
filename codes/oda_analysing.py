"""
To analysing the behavior of ODA at the interface
"""

import typing
import numpy as np
import matplotlib.pylab as plt

import logger
from get_data import GetData
import static_info as stinfo


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
        The interface location could also be caluclated dynamicly from 
        water residues
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
        self.amino_arr: np.ndarray = self.split_arr_dict['AMINO_ODN'][:-2]
        self.np_arr: np.ndarray = self.split_arr_dict['APT_COR']
        self.initiate(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """
        initiate analysing of ODA behavior
        """
        self.prepare_data(log)

        # Shift data toward the average COM of the NP
        # adjusted_oda: np.ndarray = \
            # self.shift_residues_from_np(self.amino_arr,
                                        # self.nanoparticle_disp_from_avg_com)
        # radii, molecule_counts, _, g_r = \
            # self.distribution_around_avg_np(
                # adjusted_oda, self.mean_nanop_com, 1, orginated=True)

    def prepare_data(self,
                     log: logger.logging.Logger
                     ) -> None:
        """
        1- Set NP center of mass at origin at each time frame
        2- Shift AMINO group accordingly
        3- Apply PBC to the AMINO group
        """
        amino_c: np.ndarray = self.amino_arr
        np_c: np.ndarray = self.np_arr
        shifted_amino: np.ndarray = self.com_to_zero(amino_c, np_c)

    def com_to_zero(self,
                    amino: np.ndarray,
                    np_arr: np.ndarray
                    ) -> np.ndarray:
        """subtract the np com from all the AMINO groups"""
        shifted_aminos: np.ndarray = np.empty_like(amino)
        for i in range(amino.shape[0]):
            for j in range(3):
                shifted_aminos[i, j::3] = amino[i, j::3] - np_arr[i, j]
        return shifted_aminos

    @staticmethod
    def distribution_around_avg_np(com_aligned_residues: np.ndarray,
                                   mean_nanop_com: np.ndarray,
                                   delta_r: float,
                                   orginated: bool = False,
                                   max_radius: typing.Union[float, None] = None
                                   ) -> tuple:
        """
        Calculate the distribution of ODN molecules around the
        nanoparticle's average COM.

        Parameters:
        - com_aligned_residues: Residues where their coordinates are
        relative to the nanoparticle's COM for each frame.
        - mean_nanop_com: Average COM of the nanoparticle over all frames.
        - delta_r: Width of each annulus.
        - max_radius: Maximum distance to consider. If None, will be
          calculated based on data.

        Returns:
        - bins: A list of radii corresponding to each annulus.
        - counts: A list of counts of ODN molecules in each annulus.
        """

        if not orginated:
            distances = OdaAnalysis._get_adjusted_distance(
                com_aligned_residues, mean_nanop_com)
        else:
            # Reshaping mean_np_com to broadcast-compatible with adjusted_res
            mean_nanop_com_reshaped = \
                mean_nanop_com[np.newaxis, :].repeat(
                    com_aligned_residues.shape[1]//3, axis=0).reshape(1, -1)

            # Subtracting mean_np_com from every frame of com_aligned_residues
            origin_aligned_residues = \
                com_aligned_residues - mean_nanop_com_reshaped
            distances = \
                OdaAnalysis._get_orginated_distance(origin_aligned_residues)
        # Determine max radius if not given
        if max_radius is None:
            max_radius = np.max(distances)
        print(max_radius)
        # Create bins based on delta_r
        bins = np.arange(0, max_radius + delta_r, delta_r)

        # Histogram counts for each frame and then sum over frames
        all_counts, _ = np.histogram(distances, bins=bins)
        counts = np.sum(all_counts, axis=0)

        # Return bin centers (i.e., actual radii) and counts
        bin_centers = (bins[:-1] + bins[1:]) / 2


        # Calculate the volume of each shell
        shell_volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
        
        # RDF Calculation
        # Assuming the number of particles in the system is constant over time
        bulk_density = len(distances) / (4/3 * np.pi * max_radius**3)

        g_r = counts / (shell_volumes * bulk_density)

        return bin_centers, all_counts, counts, g_r

    @staticmethod
    def _get_adjusted_distance(com_aligned_residues: np.ndarray,
                               mean_nanop_com: np.ndarray
                               ) -> np.ndarray:
        """
        calculate the distance from NP if COM of NP is not at [0,0,0]
        """
        # Calculate squared distances from each ODN to the mean_nanop_com
        distances_squared = np.sum(
            (com_aligned_residues.reshape(
             -1,
             com_aligned_residues.shape[1]//3, 3) - mean_nanop_com) ** 2,
            axis=2)

        # Get actual distances
        return np.sqrt(distances_squared)

    @staticmethod
    def _get_orginated_distance(origin_aligned_residues: np.ndarray,
                                ) -> np.ndarray:
        """
        calculate the distance if the data is orginated to [0,0,0]
        """
        distances_squared = np.sum(origin_aligned_residues.reshape(
            -1, origin_aligned_residues.shape[1]//3, 3) ** 2, axis=2)
        return np.sqrt(distances_squared)

    @staticmethod
    def shift_residues_from_np(residues: np.ndarray,
                               np_displacement: np.ndarray
                               ) -> np.ndarray:
        """
        shift coordinates of the residues relative to the displament of
        the shift of NP from averg NP com at each frame
        """
        # Determine the number of residues
        num_residues: int = residues.shape[1] // 3

        # Reshape the displacement data
        displacement_reshaped: np.ndarray = np_displacement[:, np.newaxis, :]

        # Tile the reshaped data to match the residues shape and reshape
        displacement_tiled: np.ndarray = \
            np.tile(displacement_reshaped,
                    (1, num_residues, 1)).reshape(residues.shape)

        # Subtract
        return residues - displacement_tiled


if __name__ == "__main__":
    OdaAnalysis(log=logger.setup_logger('oda_analysis.log'))

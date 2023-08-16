"""After running get_frames.py, the script reads the pickle file and
stores its contents in an array. This data will then be used for both
plotting and analyzing purposes.
The file has the following format:
elements:
The center of the mass of all residue is calculated, then it is wrapped
back into the box, and the center of mass of the NP at that time is
subtracted from it.
Number of columns is 3(N+1) + 1
    N: number of the residues,
    and one for the center of mass of the NP at the time
    and last row for saving the label of each residue
date: NN
Update:
  Reading updated com_pickle from get_frame_mpi.py Jul 21 2023
    The array layout is as follows:
        | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ... |
         resN_x | resN_y | resN_z | odn1_x| odn1_y| odn1_z| ... odnN_z|
    number of row is:
        number of frames + 2
        The extra rows are for the type of the residue at -1 and the
        orginal ids of the residues in the traj file
        number of the columns:
        n_residues: number of the residues in solution, without residues
        in NP
        n_ODA: number oda residues
        NP_com: Center of mass of the nanoparticle
        than:
        timeframe + NP_com + nr_residues:  xyz + n_oda * xyz
             1    +   3    +  nr_residues * 3  +  n_oda * 3
    The data can be split based on the index in the last row. The index
    of the ODN heads is either 0 or the index of ODN defined in the
    stinfo. If they are zero, it is straightforward forward. If not,
    the data of the ODN should be split in half.
Update Aug 15 2023:
    split the module
    GetData class is moved to another file: get_data.py
"""


import typing
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.signal import savgol_filter  # Import the savgol_filter
import static_info as stinfo
import plot_tools
from get_data import GetData


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
        """frind the shift of com of nanoparticle from mean value at
        each time step"""
        return self.split_arr_dict['APT_COR'] - self.mean_nanop_com


class PlotOdnAnalysis(WrapData):
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

    fontsize: int = 12  # Fontsize for all in plots
    transparent: bool = False  # Save fig background

    def __init__(self) -> None:
        """
        Initialize the PlotOdnAnalysis instance.

        This constructor initializes the PlotOdnAnalysis instance by
        calling the constructor of its parent class `WrapData` and
        performing necessary calculations and analyses.
        """
        super().__init__()
        odn_arr: np.ndarray = self.split_arr_dict['ODN'][:-2]
        counts: np.ndarray  # Counts of ODN in the annuluses
        radii_distance: np.ndarray  # Real distance from nanoparticle
        counts, radii_distance = self._initiate_analyses(odn_arr)
        self._initiate_plotting(odn_arr, counts, radii_distance)

    def _initiate_analyses(self,
                           odn_arr: np.ndarray  # ODN array
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initiate ODN data analysis.

        Args:
            odn_arr (np.ndarray): The array containing ODN data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing counts
            of ODN in annuluses and corresponding radii distances.
        """
        counts: np.ndarray  # Counts of ODN in the annuluses
        radii_distance: np.ndarray  # Real distance from nanoparticle
        counts, radii_distance = \
            self.count_odn_in_annuluses(odn_arr, delta_r=5)
        return counts, radii_distance

    def _initiate_plotting(self,
                           odn_arr: np.ndarray,  # ODN array
                           counts: np.ndarray,  # Counts of ODN in the annulus
                           radii_distance: np.ndarray  # Real distance from np
                           ) -> None:
        """
        Initiate ODN data plotting.

        Args:
            odn_arr (np.ndarray): The array containing ODN data.
            counts (np.ndarray): Counts of ODN in annuluses.
            radii_distance (np.ndarray): Corresponding radii distances.
        """
        self.plot_odn_com(odn_arr)
        self.plot_smoothed_annulus_density(counts, radii_distance)
        self.plot_average_annulus_density(counts, radii_distance)

    def plot_average_annulus_density(self,
                                     counts: np.ndarray,  # Counts of ODN
                                     radii_distance: np.ndarray  # Real distanc
                                     ) -> None:
        """
        Plot the average ODN density across annuluses.

        This method creates a plot showing the average ODN density in
        different annuluses.

        Args:
            counts (np.ndarray): The ODN counts in annuluses.
            radii_distance (np.ndarray): The corresponding real distances
            from the nanoparticle.
        """
        # Create subplots for each frame
        fig_i, ax_i = plot_tools.mk_canvas((self.box_dims['x_lo'],
                                            self.box_dims['x_hi']),
                                           num_xticks=6,
                                           fsize=12)

        average_counts: np.ndarray = np.average(counts[100:], axis=0)
        smoothed_counts = \
            savgol_filter(average_counts,
                          window_length=5,
                          polyorder=3)  # Apply Savitzky-Golay smoothin
        ax_i.plot(radii_distance[:-1], smoothed_counts, label='Average', c='k')
        ax_i.set_xlabel('Distance from NP')
        ax_i.set_ylabel('ODN Count')
        ax_i.set_title('ODN Counts in Annuluses')
        # Plot vertical line at the specified x-coordinate
        ax_i.axvline(x=self.nanop_radius,
                     color='red',
                     linestyle='--',
                     label='Nanoparticle')
        ax_i.axvline(x=self.interface_locz,
                     color='b',
                     linestyle='--',
                     label='interface (average)')
        plot_tools.set_y2ticks(ax_i)
        ax_i.xaxis.grid(color='gray', linestyle=':')
        ax_i.yaxis.grid(color='gray', linestyle=':')
        plot_tools.save_close_fig(fig_i, ax_i, fname='average_odn')

    def plot_smoothed_annulus_density(self,
                                      counts: np.ndarray,  # Counts of ODN
                                      radii_distance: np.ndarray  # Real dist
                                      ) -> None:
        """
        Plot smoothed ODN density across annuluses.

        This method creates a plot showing the smoothed ODN density in
        different annuluses.

        Args:
            counts (np.ndarray): The ODN counts in annuluses.
            radii_distance (np.ndarray): The corresponding real distances
            from the nanoparticle.
        """
        # Create subplots for each frame
        fig_i, ax_i = plot_tools.mk_canvas((self.box_dims['x_lo'],
                                            self.box_dims['x_hi']),
                                           num_xticks=6,
                                           fsize=12)

        for frame in range(100, self.nr_dict['nr_frames'], 10):
            # Get indices of non-zero counts
            non_zero_indices = np.nonzero(counts[frame])
            if len(non_zero_indices[0]) > 0:
                max_window_length = len(non_zero_indices[0]) // 2
                smoothed_counts = \
                    savgol_filter(counts[frame][non_zero_indices],
                                  window_length=max_window_length,
                                  polyorder=5)  # Apply Savitzky-Golay smoothin
                ax_i.plot(non_zero_indices[0],
                          smoothed_counts,
                          label=f'Frame {frame}')
        ax_i.set_xlabel('Annulus Index')
        ax_i.set_ylabel('ODN Count')
        ax_i.set_title('ODN Counts in Annuluses')
        ax_i.legend()
        plot_tools.save_close_fig(fig_i, ax_i, fname='odn_density')

    def plot_odn_com(self, odn_arr: np.ndarray) -> None:
        """
        Plot the center of mass of ION molecules.

        This method creates scatter plots showing the movement of ION
        molecules' center of mass in different dimensions.

        Args:
            ion_arr (np.ndarray): The array of ION data.
        """
        # Create a 1x3 grid of subplots
        odn_fig, odn_axes = plt.subplots(1, 3, figsize=(18, 6))
        axis_names = ['x', 'y', 'z']
        odn_data = {
            axis: odn_arr[:, axis_idx::3] for axis_idx, axis in
            enumerate(axis_names)
            }

        for i_step in range(self.nr_dict['nr_frames']):
            for ax_idx, scatter_axes in enumerate(
               [('x', 'y'), ('y', 'z'), ('x', 'z')]):
                scatter_x_axis, scatter_y_axis = scatter_axes
                shift_x = \
                    self.shift_nanop_com[i_step][0] if \
                    scatter_x_axis == 'x' else \
                    self.shift_nanop_com[i_step][2]

                shift_y = self.shift_nanop_com[i_step][1]

                scatter_x_data = odn_data[scatter_x_axis][i_step] - shift_x
                scatter_y_data = odn_data[scatter_y_axis][i_step] - shift_y

                odn_axes[ax_idx].scatter(
                    scatter_x_data,
                    scatter_y_data,
                    s=5,
                    c='black',
                    alpha=(i_step + 1) / self.nr_dict['nr_frames']
                )
                circ = plot_tools.mk_circle(radius=self.nanop_radius,
                                            center=self.mean_nanop_com[0:2])
                odn_axes[ax_idx].add_artist(circ)
                odn_axes[ax_idx].set_aspect('equal')
                odn_axes[ax_idx].set_title(
                    f'{scatter_x_axis.upper()}-{scatter_y_axis.upper()} Plane')
                odn_axes[ax_idx].set_xlabel(
                    f'{scatter_x_axis.capitalize()} Coordinate')
                odn_axes[ax_idx].set_ylabel(
                    f'{scatter_y_axis.capitalize()} Coordinate')
                odn_axes[ax_idx].grid(True)

        plt.tight_layout()
        odn_fig.savefig('odn_com_plots.png')
        plt.close(odn_fig)

    def count_odn_in_annuluses(self,
                               odn_arr: np.ndarray,  # ODN array
                               delta_r: float  # Size of the annulus (Delta r)
                               ) -> tuple[np.ndarray, ...]:
        """
        Count the number of ODN molecules in different annuluses around
        the nanoparticle.

        This method calculates the distribution of ODN molecules in
        annuluses around the nanoparticle.
        It divides the space around the nanoparticle into concentric
        annuluses, each with a width of Δr, and counts the number of
        ODN molecules that fall within each annulus. The resulting
        counts are returned along with the real distances from the
        nanoparticle center corresponding to each annulus.

        Args:
            odn_arr (np.ndarray): An array of ODN coordinates (x, y, z)
            over time frames.
            delta_r (float): The width of each annulus.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - counts: An array of shape (num_time_frames,
                  num_annuluses) representing the counts of ODN
                  molecules in each annulus for each time frame.
                - radii_distance: An array of real distances from the
                  nanoparticle center corresponding to the boundaries
                  of each annulus.

        Example:
            If odn_arr is a 2D array with ODN coordinates over time
            rames and delta_r is 5, the function will return counts of
            ODN molecules in each annulus and the corresponding real
            distances from the nanoparticle center.
        """
        # Create radii array with steps of delta_r
        radii = np.arange(self.nanop_radius, self.box_dims['x_hi'], delta_r)
        radii_distance: np.ndarray = \
            np.linspace(self.nanop_radius, self.box_dims['x_hi'], len(radii))
        num_time_frames, num_columns = odn_arr.shape
        num_residues = num_columns // 3

        counts: np.ndarray = \
            np.zeros((num_time_frames, len(radii) - 1), dtype=int)

        for frame in range(num_time_frames):

            for residue in range(num_residues):
                distance: np.float64 = \
                    self.calculate_distance(odn_arr[frame, residue*3],
                                            odn_arr[frame, residue*3 + 1],
                                            odn_arr[frame, residue*3 + 2],
                                            self.mean_nanop_com[0],
                                            self.mean_nanop_com[1],
                                            self.mean_nanop_com[2])
                # Ensure the distance is within self.nanop_radius
                if distance >= self.nanop_radius:
                    annulus_idx = \
                        self.categorize_into_annuluses(distance, radii) - 1
                    counts[frame, annulus_idx] += 1
        return counts, radii_distance

    @staticmethod
    def categorize_into_annuluses(distances: np.float64,
                                  radii: np.ndarray
                                  ) -> np.int64:
        """
        Categorize distances into annuluses based on their magnitude.

        This method categorizes distances from the nanoparticle center
        into different annuluses based on their magnitude. Each distance
        is assigned to an annulus based on the defined radii intervals.
        Distances falling within a particular interval [r_i, r_{i+1})
        are assigned to the i-th annulus.

        Args:
            distances (np.float64): An array of distances to be
            categorized.
            radii (np.ndarray): An array of radii defining the boundaries
            of annuluses.

        Returns:
            np.int64: An array of indices indicating the annulus to
            which each distance belongs.

        Example:
            If radii = [2, 5, 7, 10] and distances = [3.5, 8.2, 4.0],
            then the function will return [1, 3, 2], indicating that
            the first distance falls within the interval [2, 5), the
            second distance falls within the interval [7, 10), and the
            third distance falls within the interval [5, 7).
        """
        return np.digitize(distances, radii)

    @staticmethod
    def calculate_distance(x_i: float, y_i: float, z_i: float,  # First point
                           x_2: float, y_2: float, z_2: float   # Second point
                           ) -> np.float64:
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt((x_2 - x_i)**2 + (y_2 - y_i)**2 + (z_2 - z_i)**2)


class PlotIonAnalysis(WrapData):
    """
    A class for analyzing ion data and creating ion density plots.

    This class inherits from the `WrapData` class and provides methods
    to analyze ion data and generate plots related to ion densities.

    Attributes:
        fontsize (int): Font size for all plots.
        transparent (bool): Whether to save fig background as transparent.

    Methods:
        __init__(): Initialize the PlotIonAnalysis instance.
        initiate_ion_analysis(ion_arr: np.ndarray, delta_z: float)
            -> Tuple[np.ndarray, np.ndarray]:
            Initiate ion data analysis.
        count_ions_in_slabs(ion_arr: np.ndarray, delta_z: float) ->
            Tuple[np.ndarray, np.ndarray]:
            Count the number of ions in different slabs.
        categorize_into_slabs(z_coord: float, slab_areas: np.ndarray)
            -> np.int64:
            Categorize z-coordinate into slabs.
        initiate_ion_plotting(counts: np.ndarray, slab_areas: np.ndarray)
            -> None:
            Initiate ion data plotting.
        plot_smoothed_ion_density(counts: np.ndarray, slab_areas:
            np.ndarray) -> None:
            Plot ion density in different slabs.
    """
    fontsize: int = 12  # Fontsize for all in plots
    transparent: bool = False  # Save fig background

    def __init__(self) -> None:
        super().__init__()
        ion_arr: np.ndarray = self.split_arr_dict['CLA'][:-2]
        counts, slab_areas = self.initiate_ion_analysis(ion_arr, delta_z=5)
        self.initiate_ion_plotting(ion_arr, counts, slab_areas)

    def initiate_ion_analysis(self,
                              ion_arr: np.ndarray,
                              delta_z: float
                              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initiate ion data analysis.

        Args:
            ion_arr (np.ndarray): Array containing ion coordinates (x, y, z).
            delta_z (float): Thickness of each slab.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing ion counts
            in each slab and corresponding slab areas.
        """
        counts, slab_areas = self.count_ions_in_slabs(ion_arr, delta_z)
        return counts, slab_areas

    def count_ions_in_slabs(self,
                            ion_arr: np.ndarray,
                            delta_z: float
                            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Count the number of ions in different slabs.

        This method calculates the distribution of ions in slabs along
        the z-coordinate. It divides the z-range of the simulation box
        into consecutive slabs of thickness `delta_z` and counts the
        number of ions that fall within each slab. The resulting counts
        are returned along with the corresponding z-coordinates of slab
        boundaries.

        Args:
            ion_arr (np.ndarray): Array containing ion coordinates
            (x, y, z).
            delta_z (float): Thickness of each slab.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - counts: An array of shape (num_time_frames, num_slabs)
                  representing the counts of ions in each slab for each
                  time frame.
                - slab_areas: An array of z-coordinates representing the
                  boundaries of each slab.

        Example:
            If ion_arr is a 2D array with ion coordinates over time
            frames and delta_z is 5, the function will return counts
            of ions in each slab and the corresponding slab boundaries.
        """
        slab_areas = \
            np.arange(self.box_dims['z_lo'], self.box_dims['z_hi'], delta_z)
        num_time_frames, num_ions = ion_arr.shape
        num_slabs = len(slab_areas)

        counts = np.zeros((num_time_frames, num_slabs), dtype=int)

        for frame in range(num_time_frames):
            for ion in range(num_ions//3):
                z_coord = ion_arr[frame, ion*3 + 2]
                slab_idx = \
                    self.categorize_into_slabs(z_coord, slab_areas) - 1
                counts[frame, slab_idx] += 1

        return counts, slab_areas

    def categorize_into_slabs(self,
                              z_coord: float,
                              slab_areas: np.ndarray
                              ) -> np.int64:
        """
        Categorize z-coordinate into slabs.

        This method categorizes z-coordinates of ions into different
        slabs based on their value. Each z-coordinate is assigned to
        a specific slab based on the slab areas provided.

        Args:
            z_coord (float): Z-coordinate of an ion.
            slab_areas (np.ndarray): An array of z-coordinates defining
            the boundaries of slabs.

        Returns:
            np.int64: Index indicating the slab to which the z-coordinate
            belongs.

        Example:
            If slab_areas = [0, 5, 10] and z_coord = 8.2, then the
            function will return 2, indicating that the z-coordinate
            falls within the second slab [5, 10).
        """
        return np.digitize(z_coord, slab_areas)

    def initiate_ion_plotting(self,
                              ion_arr: np.ndarray,
                              counts: np.ndarray,
                              slab_areas: np.ndarray
                              ) -> None:
        """
        Initiate ion data plotting.

        Args:
            counts (np.ndarray): Ion counts in different slabs.
            slab_areas (np.ndarray): Z-coordinates representing slab
            boundaries.
        """
        self.plot_ion_com(ion_arr)
        self.plot_smoothed_ion_density(counts, slab_areas)
        self.plot_average_slabs_density(counts, slab_areas)

    def plot_average_slabs_density(self,
                                   counts: np.ndarray,  # Counts of ION
                                   radii_distance: np.ndarray  # Real distanc
                                   ) -> None:
        """
        Plot the average ION density across annuluses.

        This method creates a plot showing the average ION density in
        different annuluses.

        Args:
            counts (np.ndarray): The ION counts in annuluses.
            radii_distance (np.ndarray): The corresponding real distances
            from the nanoparticle.
        """
        # Create subplots for each frame
        fig_i, ax_i = plot_tools.mk_canvas((self.box_dims['x_lo'],
                                            self.box_dims['x_hi']),
                                           num_xticks=6,
                                           fsize=12)

        average_counts: np.ndarray = np.average(counts[100:], axis=0)
        smoothed_counts = \
            savgol_filter(average_counts,
                          window_length=5,
                          polyorder=3)  # Apply Savitzky-Golay smoothin
        ax_i.plot(radii_distance, smoothed_counts, label='Average', c='k')
        ax_i = self.append_vlines_span_to_density(ax_i)
        plot_tools.save_close_fig(fig_i, ax_i, fname='average_ion')

    def plot_smoothed_ion_density(self,
                                  counts: np.ndarray,
                                  slab_areas: np.ndarray
                                  ) -> None:
        """
        Plot ion density in different slabs.

        This method creates a plot showing the ion density distribution
        in different slabs along the z-coordinate.

        Args:
            counts (np.ndarray): Ion counts in different slabs.
            slab_areas (np.ndarray): Z-coordinates representing slab
            boundaries.
        """
        fig_i, ax_i = plot_tools.mk_canvas((self.box_dims['x_lo'],
                                            self.box_dims['x_hi']),
                                           num_xticks=6,
                                           fsize=12)
        for frame in range(100, self.nr_dict['nr_frames'], 1):
            smoothed_counts = \
                savgol_filter(counts[frame], window_length=5, polyorder=3)
            ax_i.plot(slab_areas,
                      smoothed_counts,
                      lw=0.2,
                      color='k',
                      alpha=(frame-99)/(self.nr_dict['nr_frames']-100))
        ax_i = self.append_vlines_span_to_density(ax_i)
        plot_tools.save_close_fig(
            fig_i, ax_i, fname='ion_density', legend=False)

    def append_vlines_span_to_density(self,
                                      ax_i: plt.axes
                                      ) -> plt.axes:
        """append vlines and stuff to the axis and retrun it"""
        ax_i.axvspan(self.mean_nanop_com[2] - self.nanop_radius,
                     self.mean_nanop_com[2] + self.nanop_radius,
                     color='red',
                     edgecolor=None,
                     linestyle='',
                     alpha=0.15)
        # Plot vertical line at the specified x-coordinate
        ax_i.axvline(x=self.mean_nanop_com[2],
                     color='red',
                     linestyle='--',
                     label='NP z of average COM')
        ax_i.axvline(x=self.interface_locz,
                     color='b',
                     linestyle='--',
                     label='interface (average)')
        ax_i.set_xlabel('distance in z-direction')
        ax_i.set_ylabel('ION Count')
        ax_i.set_title('ION Counts in slabs')
        plot_tools.set_y2ticks(ax_i)
        ax_i.xaxis.grid(color='gray', linestyle=':')
        ax_i.yaxis.grid(color='gray', linestyle=':')
        return ax_i

    def plot_ion_com(self, ion_arr: np.ndarray) -> None:
        """
        Plot the center of mass of ION molecules.

        This method creates scatter plots showing the movement of ION
        molecules' center of mass in different dimensions.

        Args:
            ion_arr (np.ndarray): The array of ION data.
        """
        # Create a 1x3 grid of subplots
        ion_fig, ion_axes = plt.subplots(1, 3, figsize=(18, 6))
        axis_names = ['x', 'y', 'z']
        ion_data = {
            axis: ion_arr[:, axis_idx::3] for axis_idx, axis in
            enumerate(axis_names)
            }

        for i_step in range(self.nr_dict['nr_frames']):
            for ax_idx, scatter_axes in enumerate(
               [('x', 'y'), ('y', 'z'), ('x', 'z')]):
                scatter_x_axis, scatter_y_axis = scatter_axes
                shift_x = \
                    self.shift_nanop_com[i_step][0] if \
                    scatter_x_axis == 'x' else \
                    self.shift_nanop_com[i_step][2]

                shift_y = self.shift_nanop_com[i_step][1]

                scatter_x_data = ion_data[scatter_x_axis][i_step] - shift_x
                scatter_y_data = ion_data[scatter_y_axis][i_step] - shift_y

                ion_axes[ax_idx].scatter(
                    scatter_x_data,
                    scatter_y_data,
                    s=5,
                    c='black',
                    alpha=(i_step + 1) / self.nr_dict['nr_frames']
                )
                circ = plot_tools.mk_circle(radius=self.nanop_radius,
                                            center=self.mean_nanop_com[0:2])
                ion_axes[ax_idx].add_artist(circ)
                ion_axes[ax_idx].set_aspect('equal')
                ion_axes[ax_idx].set_title(
                    f'{scatter_x_axis.upper()}-{scatter_y_axis.upper()} Plane')
                ion_axes[ax_idx].set_xlabel(
                    f'{scatter_x_axis.capitalize()} Coordinate')
                ion_axes[ax_idx].set_ylabel(
                    f'{scatter_y_axis.capitalize()} Coordinate')
                ion_axes[ax_idx].grid(True)

        plt.tight_layout()
        ion_fig.savefig('ion_com_plots.png')
        plt.close(ion_fig)


class PlotNpAnalysis(WrapData):
    """
    Analyzes and plots the behavior of nanoparticles.

    This class inherits from WrapData and provides methods for
    analyzing and plotting the behavior of nanoparticles' center of
    mass (COM) data. It creates visualizations with shaded regions and
    insets for detailed information.

    Attributes:
        fontsize (int): Font size for all plots.
        transparent (bool): Option to save the figure with a
        transparent background.

    Methods:
        __init__(): Constructor to initialize the class and initiate
                    plotting.
        _initiate_plotting(): Initiate nanoparticle center of mass
                              plotting.
        _add_background_shading(): Add background shading to the main
                                   plot.
        _add_radius_shading(): Add radius shading to the main plot.
        _plot_inset(): Plot an inset showing center of mass details.
        _finalize_plot(): Finalize the nanoparticle center of mass
                          plot.
    """

    fontsize: int = 12
    transparent: bool = False

    def __init__(self) -> None:
        super().__init__()
        nanop_arr: np.ndarray = self.split_arr_dict['APT_COR'][:-2]
        self._initiate_plotting(nanop_arr)

    def _initiate_plotting(self,
                           nanop_arr: np.ndarray
                           ) -> None:
        """
        Initiate the nanoparticle center of mass plotting.

        This method sets up the figure, axes, and plots the
        nanoparticle center of mass data along with shading for
        background and radius information. It also calls the method to
        add an inset plot.
        """
        fig_i, ax_i = plot_tools.mk_canvas(
            (self.box_dims['x_lo'], self.box_dims['x_hi']),
            num_xticks=6,
            fsize=12
        )
        ax_i.plot(nanop_arr[:, 2], c='k', label='COM of NP')
        self._add_background_shading(ax_i)
        self._add_radius_shading(ax_i)
        inset_ax = self._plot_inset(nanop_arr)
        self._finalize_plot(fig_i, ax_i, inset_ax)

    def _add_background_shading(self,
                                ax_i: plt.axes
                                ) -> None:
        """
        Add background shading to the plot.

        This method adds shading to the background of the main plot
        representing different regions.

        Args:
            ax_i (plt.axes): The main axes of the plot.

        Returns:
            None
        """
        ax_i.axhspan(
            ymin=self.box_dims['z_lo'],
            ymax=self.interface_locz,
            xmin=0,
            xmax=200,
            color='blue',
            alpha=0.05
        )
        ax_i.axhspan(
            ymin=self.interface_locz,
            ymax=self.box_dims['z_hi'],
            xmin=0,
            xmax=200,
            color='yellow',
            alpha=0.05
        )

    def _add_radius_shading(self,
                            ax_i: plt.axes
                            ) -> None:
        """
        Add radius shading to the plot.

        This method adds shading to the plot representing the radius
        of the nanoparticle center of mass.

        Args:
            ax_i (plt.axes): The main axes of the plot.

        Returns:
            None
        """
        ax_i.axhspan(
            ymin=self.mean_nanop_com[2] - self.nanop_radius,
            ymax=self.mean_nanop_com[2] + self.nanop_radius,
            edgecolor='red',  # Color of the hatch lines
            facecolor='none',
            hatch='////',  # Set the hatch patter
            linestyle='',
            alpha=0.3,
            label='Nanoparticle Radius'
        )

    def _plot_inset(self,
                    nanop_arr: np.ndarray
                    ) -> plt.axes:
        """
        Plot an inset showing center of mass details.

        This method adds an inset plot to the main plot, showing
        details of the nanoparticle center of mass data.

        Args:
            nanop_arr (np.ndarray): Nanoparticle center of mass data.

        Returns:
            plt.axes: The axes of the inset plot.
        """
        left, bottom, width, height = [0.23, 0.62, 0.65, 0.23]
        inset_ax = plt.axes([left, bottom, width, height])
        inset_ax.plot(nanop_arr[:, 2], c='k')
        inset_ax.axhline(
            self.mean_nanop_com[2],
            color='r',
            ls='--',
            lw=1,
            label=f'Mean of COM: {self.mean_nanop_com[2]:.2f}'
        )
        inset_ax.legend(fontsize=9)
        return inset_ax

    def _finalize_plot(self,
                       fig: plt.figure,
                       ax_i: plt.axes,
                       legend: typing.Optional[bool] = True,
                       loc: typing.Optional[str] = 'lower left'
                       ) -> None:
        """
        Finalize the nanoparticle center of mass plot.

        This method adds final touches to the main plot, including
        labels, legend, and saves the figure.

        Args:
            fig (plt.figure): The main figure.
            ax_i (plt.axes): The main axes of the plot.
            legend (Optional[bool]): Whether to show the legend.
            loc (Optional[str]): Location of the legend.

        Returns:
            None
        """
        ax_i.set_xlabel('Time Frame Index (*0.1 [ns])')
        ax_i.set_ylabel('Z Coordinate [A]')
        plot_tools.save_close_fig(
            fig, ax_i, fname='nanop_com', legend=legend, loc=loc)


class PlotWaterAnalysis(WrapData):
    """
    To analysis water
    """
    fontsize: int = 12  # Fontsize for all in plots
    transparent: bool = False  # Save fig background

    def __init__(self) -> None:
        super().__init__()
        sol_arr: np.ndarray = self.split_arr_dict['SOL'][:-2]
        clean_arr = self.initiate_sol_analysis(sol_arr)
        self.initiate_sol_plotting(clean_arr)

    def initiate_sol_analysis(self,
                              sol_arr: np.ndarray
                              ) -> np.ndarray:
        """do some stuff"""
        return self.clean_sol_data(sol_arr)

    def clean_sol_data(self,
                       sol_arr: np.ndarray
                       ) -> np.ndarray:
        """clean data by removing water at the edge of the box"""
        treshhold: float = 10
        x_min: float = self.box_dims['x_lo'] + treshhold
        x_max: float = self.box_dims['x_hi'] - treshhold
        y_min: float = self.box_dims['y_lo'] + treshhold
        y_max: float = self.box_dims['y_hi']
        z_min: float = self.box_dims['z_lo'] + treshhold
        z_max: float =  150
        # Create boolean masks for each condition
        x_mask = (sol_arr[:, 0] >= x_min) & (sol_arr[:, 0] <= x_max)
        y_mask = (sol_arr[:, 1] >= y_min) & (sol_arr[:, 1] <= y_max)
        z_mask = (sol_arr[:, 2] >= z_min) & (sol_arr[:, 2] <= z_max)

        # Combine the masks to get the final mask
        final_mask = x_mask & y_mask & z_mask


        # Set values outside the threshold to zero
        clean_arr = sol_arr.copy()
        clean_arr[final_mask] = 0
        
        return clean_arr

    def initiate_sol_plotting(self,
                              sol_arr: np.ndarray
                              ) -> None:
        """plot sol"""
        self.plot_sol_com(sol_arr)

    def plot_sol_com(self,
                     sol_arr: np.ndarray) -> None:
        """
        Plot the center of mass of SOL molecules.

        This method creates scatter plots showing the movement of SOL
        molecules' center of mass in different dimensions.

        Args:
            sol_arr (np.ndarray): The array of SOL data.
        """
        # Create a 1x3 grid of subplots
        sol_fig, sol_axes = plt.subplots(1, 3, figsize=(18, 6))
        axis_names = ['x', 'y', 'z']
        sol_data = {
            axis: sol_arr[:, axis_idx::3] for axis_idx, axis in
            enumerate(axis_names)
            }

        for i_step in range(self.nr_dict['nr_frames']):
            for ax_idx, scatter_axes in enumerate(
               [('x', 'y'), ('y', 'z'), ('x', 'z')]):
                scatter_x_axis, scatter_y_axis = scatter_axes
                shift_x = \
                    self.shift_nanop_com[i_step][0] if \
                    scatter_x_axis == 'x' else \
                    self.shift_nanop_com[i_step][2]

                shift_y = self.shift_nanop_com[i_step][1]

                scatter_x_data = sol_data[scatter_x_axis][i_step] #- shift_x
                scatter_y_data = sol_data[scatter_y_axis][i_step] #- shift_y
                sol_axes[ax_idx].scatter(
                    scatter_x_data,
                    scatter_y_data,
                    s=5,
                    c='black',
                    alpha=(i_step + 1) / self.nr_dict['nr_frames']
                )
                circ = plot_tools.mk_circle(radius=self.nanop_radius,
                                            center=self.mean_nanop_com[0:2])
                sol_axes[ax_idx].add_artist(circ)
                sol_axes[ax_idx].set_aspect('equal')
                sol_axes[ax_idx].set_title(
                    f'{scatter_x_axis.upper()}-{scatter_y_axis.upper()} Plane')
                sol_axes[ax_idx].set_xlabel(
                    f'{scatter_x_axis.capitalize()} Coordinate')
                sol_axes[ax_idx].set_ylabel(
                    f'{scatter_y_axis.capitalize()} Coordinate')
                sol_axes[ax_idx].grid(True)

        plt.tight_layout()
        sol_fig.savefig('sol_com_plots.png')
        plt.close(sol_fig)


if __name__ == '__main__':
    PlotOdnAnalysis()
    # PlotIonAnalysis()
    # PlotNpAnalysis()
    # PlotWaterAnalysis()

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


import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.signal import savgol_filter  # Import the savgol_filter
import static_info as stinfo
import plot_interface_z as plt_z
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
        self.interface_locz = 107.7
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
        self.plot_odn(odn_arr)
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
        ax_i.plot(radii_distance[:-1], smoothed_counts, label='Average')
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
        ax_i.legend()
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

    def plot_odn(self,
                 odn_arr: np.ndarray  # ODN array
                 ) -> None:
        """
        Plot the center of mass of ODN molecules.

        This method creates a plot showing the movement of ODN
        molecules' center of mass.

        Args:
            odn_arr (np.ndarray): The array of ODN data.
        """
        odn_fig, odn_ax = plt.subplots()
        axis_names = ['x', 'y', 'z']
        odn_data: dict[str, np.ndarray] = {}
        for axis_idx, axis in enumerate(axis_names):
            odn_data[axis] = odn_arr[:, axis_idx::3]
        # Find the ODN at the interface
        mask = (odn_data['z'] < self.interface_locz+10) & \
               (odn_data['z'] > self.interface_locz-10)
        # Get the indices where the mask is True for each row
        indices: list[np.ndarray] = \
            [np.where(row_mask)[0] for row_mask in mask]
        for i_step in range(self.nr_dict['nr_frames']):
            x_data = odn_data['x'][i_step][indices[i_step]] - \
                self.shift_nanop_com[i_step][0]
            y_data = odn_data['y'][i_step][indices[i_step]] - \
                self.shift_nanop_com[i_step][1]
            odn_ax.scatter(x_data,
                           y_data,
                           s=5,
                           c='black',
                           alpha=(i_step+1)/self.nr_dict['nr_frames'])
        circ: matplotlib.patches.Circle = \
            plot_tools.mk_circle(radius=self.nanop_radius,
                                 center=self.mean_nanop_com[0:2])
        odn_ax.add_artist(circ)
        plt.axis('equal')
        odn_fig.savefig('test2_odn.png')

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
        plot_smoothed_ion_density(counts: np.ndarray, slab_areas: np.ndarray) -> None:
            Plot ion density in different slabs.
    """
    fontsize: int = 12  # Fontsize for all in plots
    transparent: bool = False  # Save fig background

    def __init__(self) -> None:
        super().__init__()
        ion_arr: np.ndarray = self.split_arr_dict['CLA'][:-2]
        counts, slab_areas = self.initiate_ion_analysis(ion_arr, delta_z=5)
        self.initiate_ion_plotting(counts, slab_areas)

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
        self.plot_smoothed_ion_density(counts, slab_areas)

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
        fig_i, ax_i = plot_tools.mk_canvas(
            (self.box_dims['x_lo'], self.box_dims['x_hi']), num_xticks=6)
        for frame in range(100, self.nr_dict['nr_frames'], 10):
            smoothed_counts = \
                savgol_filter(counts[frame], window_length=5, polyorder=3)
            ax_i.plot(slab_areas, smoothed_counts, label=f'Frame {frame}')
        ax_i.set_xlabel('Z Coordinate')
        ax_i.set_ylabel('Ion Count')
        ax_i.set_title('Ion Counts in Slabs')
        ax_i.legend()
        plot_tools.save_close_fig(fig_i, ax_i, fname='ion_density')


class PlotCom(GetData):
    """
    Reading the center of mass file, the name is set static_info.py
    Attributes from parent class:
        f_name (str): The filename of the input pickle file.
        split_arr_dict (dict[str, np.ndarray]): A dictionary containing
            split data arrays organized by residue types.
        nr_dict (dict[str, int]): A dictionary containing numerical
            information about the data, including the number of time
            frames and the number of residues for each type.
        box_dims (dict[str, float]): A dictionary containing the
            the dimensions of the box
    """

    def __init__(self) -> None:
        super().__init__()
        self.f_name: str = stinfo.files['com_pickle']
        self.plot_interface: bool = False  # If plot and save water
        self.plot_com()

    def plot_com(self) -> None:
        """plot data based on what needed.
           plot com (center of mass) of the ODA and ION separately.
           plot topology if water at the interface.
        """
        ax_com = plt.gca()
        x_indices: range  # Range of the indices
        y_indices: range  # Range of the indices
        z_indices: range  # Range of the indices
        interface_locz: list[tuple[float, float]] = []  # Z mean & std of water
        for res in ['ODN', 'CLA', 'SOL']:
            res_arr: np.ndarray = self.__get_residue(res)
            x_indices, y_indices, z_indices = self.__get_res_xyz(res_arr)
            for i in range(self.nr_dict['nr_frames']):
                if res in ['ODN']:
                    x_data, y_data, _ = \
                        self.__get_interface_oda(res_arr[i, x_indices],
                                                 res_arr[i, y_indices],
                                                 res_arr[i, z_indices],
                                                 res)
                    ax_com.scatter(x_data, y_data, s=5, c='black',
                                   alpha=(i+1)/self.nr_dict['nr_frames'])
                if res in ['SOL']:
                    x_surf, y_surf, z_surf = \
                        self.__plot_water_surface(res_arr[i, x_indices],
                                                  res_arr[i, y_indices],
                                                  res_arr[i, z_indices],
                                                  i)
                    interface_locz.append(
                        self.get_interface_loc(x_surf, y_surf, z_surf))
            if res in ['ODN', 'CLA']:
                self.__plot_odn_com(ax_com, res)
        # Plot interface based on the z location
        plt_z.PlotInterfaceZ(interface_locz)

    def find_mean_of_np_com(self) -> np.ndarray:
        """find mean of the nanoparticle center of mass"""
        return np.mean(self.split_arr_dict['APT_COR'], axis=0)

    def find_np_shift_from_mean(self,
                                apt_cor_mean: np.ndarray
                                ) -> np.ndarray:
        """frind the shift of com of nanoparticle from mean value at
        each time step"""
        return self.split_arr_dict['APT_COR'] - apt_cor_mean

    def get_interface_loc(self,
                          x_data: np.ndarray,  # x component of water interface
                          y_data: np.ndarray,  # y component of water interface
                          z_data: np.ndarray,  # z component of water interface
                          ) -> tuple[float, float]:
        """To locate the surface interface, look for water molecules
        at the interface and not above or below the NP. One way to do
        this is by calculating the average z components of the interface.
        Although there may be more accurate methods, this is the most
        straightforward and practical."""
        lengths: np.ndarray = self.__calculate_lengths(x_data, y_data)
        ind_outsid: np.ndarray = np.where(lengths > stinfo.np_info['radius'])
        z_outsid: np.ndarray = z_data[ind_outsid]
        # Calculate the confidence interval
        mean = np.mean(z_outsid)  # Mean of the data
        std_error = np.std(z_outsid)
        return mean, std_error

    def __plot_water_surface(self,
                             x_data_all: np.ndarray,  # All x values for sol
                             y_data_all: np.ndarray,  # All y values for sol
                             z_data_all: np.ndarray,  # All z values for sol
                             i_time: int  # Number of the frame
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """find the surface water residues com and plot them for each
        timestep"""
        # Getting data for water on surface
        x_surf: np.ndarray  # x component of the surface
        y_surf: np.ndarray  # y component of the surface
        z_surf: np.ndarray  # z component of the surface
        x_surf, y_surf, z_surf = \
            self.__get_interface(x_data_all, y_data_all, z_data_all)
        if self.plot_interface:
            # Scatter the data
            cmap: str = 'viridis'  # Color for the maping
            fig, ax_surf = plt.subplots()
            ax_surf.scatter(x_surf, y_surf, c=z_surf, cmap=cmap,
                            label=f"frame: {i_time}")
            # Get equal axises
            ax_surf.set_aspect('equal')
            # Create a ScalarMappable object for the color mapping
            smap = plt.cm.ScalarMappable(cmap=cmap)
            smap.set_array(z_surf)
            cbar = fig.colorbar(smap, ax=ax_surf)
            # set the axis
            ax_surf.set_xlim(self.box_dims['x_lo'], self.box_dims['x_hi'])
            ax_surf.set_ylim(self.box_dims['y_lo'], self.box_dims['y_hi'])
            # Labels
            cbar.set_label('z [A]')
            ax_surf.set_xlabel('x [A]')
            ax_surf.set_ylabel('y [A]')
            # I wanna circle around the NP position
            circle: bool = True  # If want to add circle
            if circle:
                # Get the current axes and add the circle to the plot
                ax_surf.add_artist(self.__mk_circle())
            # Show legend
            plt.legend(loc='lower left')
            # Set the name of the ouput file
            pname: str = f'water_surface_frame_{i_time}.png'
            plt.savefig(pname, bbox_inches='tight', transparent=False)
            plt.close(fig)
        return x_surf, y_surf, z_surf

    def __get_interface(self,
                        x_data_all: np.ndarray,  # All the x values for sol
                        y_data_all: np.ndarray,  # All the y values for sol
                        z_data_all: np.ndarray,  # All the z values for sol
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """find and retrun water residue at the interface.
        Using the grid meshes in the x and y directions, the water_com
        in each grid with the highest z value is returned.
        """
        z_treshholf: float  # To drop water below some ratio of the NP radius
        z_treshholf = stinfo.np_info['radius'] * 1.2
        mesh_size: np.float64   # Size of the grid
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        x_data: np.ndarray  # Component of the points in real water phase
        y_data: np.ndarray  # Component of the points in real water phase
        z_data: np.ndarray  # Component of the points in real water phase
        x_data, y_data, z_data = \
            self.__get_in_box(x_data_all, y_data_all, z_data_all)
        x_mesh, y_mesh, mesh_size = self.__get_grid_xy(x_data, y_data)
        max_z_index: list[int] = []  # Index of the max value at each grid
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                # Define the boundaries of the current mesh element
                x_min_mesh, x_max_mesh = x_mesh[i, j], x_mesh[i, j] + mesh_size
                y_min_mesh, y_max_mesh = y_mesh[i, j], y_mesh[i, j] + mesh_size

                # Select atoms within the current mesh element based on X and Y
                ind_in_mesh = np.where((x_data >= x_min_mesh) &
                                       (x_data < x_max_mesh) &
                                       (y_data >= y_min_mesh) &
                                       (y_data < y_max_mesh) &
                                       (z_data >= -z_treshholf))
                if len(ind_in_mesh[0]) > 0:
                    max_z = np.argmax(z_data[ind_in_mesh])
                    max_z_index.append(ind_in_mesh[0][max_z])
        return x_data[max_z_index], y_data[max_z_index], z_data[max_z_index]

    def __get_residue(self,
                      res: str,  # Name of the residue to get the data,
                      ) -> np.ndarray:
        """return the info for the selected residue"""
        return self.split_arr_dict[res]

    def __plot_odn_com(self,
                       ax_com: matplotlib.axes,  # The center of mass data
                       res: str  # Name of the residue to save file
                       ) -> None:
        """plot and save the COM of the ODA"""
        circle: bool = True  # Add circle to the image
        if circle:
            # Create a circle with origin at (0, 0) and radius of np
            circ: matplotlib.patches.Circle = self.__mk_circle()
            ax_com.add_artist(circ)

        # Get the current axes and add the circle to the plot
        # Set the aspect ratio to 'equal'
        ax_com.set_aspect('equal')
        ax_com.set_xlim(self.box_dims['x_lo'], self.box_dims['x_hi'])
        ax_com.set_ylim(self.box_dims['y_lo'], self.box_dims['y_hi'])

        ax_com.set_xlabel("x component [A]")
        ax_com.set_ylabel("y component [A]")
        plt.title(f'center of mass of {res} around NP')
        pname: str  # Name of the output file
        pname = f'{res}_com.png'
        plt.savefig(pname, bbox_inches='tight', transparent=False)

    @staticmethod
    def __calculate_lengths(x_data: np.ndarray,  # x array of interface water
                            y_data: np.ndarray,  # y array of interface water
                            ) -> np.ndarray:
        """return the length of vectors from center to the water
        molecules"""
        vectors: np.ndarray = np.column_stack((x_data, y_data))
        lengths: np.ndarray = np.linalg.norm(vectors, axis=1)
        return lengths

    @staticmethod
    def __get_grid_xy(x_data: np.ndarray,  # x component of the coms
                      y_data: np.ndarray,  # y component of the coms
                      ) -> tuple[np.ndarray, np.ndarray, np.float64]:
        """return the mesh grid for the xy of sol"""
        x_min: np.float64 = np.min(x_data)
        y_min: np.float64 = np.min(y_data)
        x_max: np.float64 = np.max(x_data)
        y_max: np.float64 = np.max(y_data)
        mesh_size: np.float64 = (x_max-x_min)/100.
        x_mesh: np.ndarray  # Mesh grid in x and y
        y_mesh: np.ndarray  # Mesh grid in x and y
        x_mesh, y_mesh = np.meshgrid(
            np.arange(x_min, x_max + mesh_size, mesh_size),
            np.arange(y_min, y_max + mesh_size, mesh_size))
        return x_mesh, y_mesh, mesh_size

    @staticmethod
    def __get_in_box(x_data_all: np.ndarray,  # All the x values for sol
                     y_data_all: np.ndarray,  # All the y values for sol
                     z_data_all: np.ndarray,  # All the z values for sol
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return the index of the residues in the box, and not on the
        top of the oil pahse.
        Since the com is moved to zero, and water is under water, below
        zero, the treshhold is small number.
        """
        sol_treshhold: float  # The value above that not needed
        sol_treshhold = stinfo.np_info['radius'] * 1
        index_in_box: np.ndarray  # Index of the residues inside, not out
        index_in_box = np.where(z_data_all < sol_treshhold)[0]
        x_data_in_box = x_data_all[index_in_box]
        y_data_in_box = y_data_all[index_in_box]
        z_data_in_box = z_data_all[index_in_box]
        return x_data_in_box, y_data_in_box, z_data_in_box

    @staticmethod
    def __get_res_xyz(res_arr: np.ndarray,  # All the times
                      ) -> tuple[range, range, range]:
        """return the x, y, and z data for the residues"""
        x_indices = range(0, res_arr.shape[1], 3)
        y_indices = range(1, res_arr.shape[1], 3)
        z_indices = range(2, res_arr.shape[1], 3)
        return x_indices, y_indices, z_indices

    @staticmethod
    def __get_interface_oda(x_data: np.ndarray,  # All the x values for the oda
                            y_data: np.ndarray,  # All the x values for the oda
                            z_data: np.ndarray,  # All the x values for the oda
                            res: str  # Name of the residue
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get xyz for all the oda at the interface and or ion in the bolck"""
        interface_z: float = 107.7  # should be calculated, interface treshhold
        if res == 'ODN':
            inface_indx = np.where(z_data > interface_z)[0]  # At interface
        elif res == 'CLA':
            inface_indx = np.where(z_data < interface_z)[0]  # At interface
        return x_data[inface_indx], y_data[inface_indx], z_data[inface_indx]

    @staticmethod
    def __mk_circle(center=(0, 0)) -> matplotlib.patches.Circle:
        radius = stinfo.np_info['radius']
        circle = plt.Circle(center,
                            radius,
                            color='red',
                            linestyle='dashed',
                            fill=False, alpha=1)
        return circle


if __name__ == '__main__':
    # PlotCom()
    # PlotOdnAnalysis()
    PlotIonAnalysis()

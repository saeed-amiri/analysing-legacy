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
import static_info as stinfo
import plot_interface_z as plt_z
from get_data import GetData

class WrapPlots(GetData):
    """
    Get data and call other classes to analysis and plot them.
    Before that, some calculateion must be done
    """

    mean_nanop_com: np.ndarray  # Average of the nanoparticle COM over times
    shift_nanop_com: np.ndarray  # Shift of COM at each time from average
    interface_locz: float  # Z location of the interface

    def __init__(self) -> None:
        super().__init__()
        self.set_constants()
        self.initiate_calc()
        print(self.__dict__)

    def set_constants(self) -> None:
        """
        Set the constants for all the uses
        """
        self.interface_locz = 107.7

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


class PlotOdnAnalysis(WrapPlots):
    """
    Analysing ODN data and plot them in files
    """
    def __init__(self) -> None:
        super().__init__()


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
        nanop_center: np.ndarray = self.find_mean_of_np_com()
        nanop_shift: np.ndarray = self.find_np_shift_from_mean(nanop_center)
        self.plot_odn(self.split_arr_dict['ODN'],
                      interface_loc=107.7,
                      nanop_shift=nanop_shift,
                      nanop_center=nanop_center)
        self.plot_annulus_dendity(self.split_arr_dict['ODN'][:-2,:],
                                  interface_loc=107.7,
                                  nanop_shift=nanop_shift,
                                  nanop_center=nanop_center)
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

    def plot_annulus_dendity(self,
                             res_arr: np.ndarray,  # ODN array
                             interface_loc: float,  # Location of interface
                             nanop_shift: np.ndarray,  # Shift of np from ave
                             nanop_center: np.ndarray
                             ) -> None:
        """plot density of the odn around np in"""
        self.calculate_odn_density(res_arr,
                                   interface_loc,
                                   nanop_shift,
                                   nanop_center)

    def calculate_odn_density(self,
                              res_arr: np.ndarray,  # ODN array
                              interface_loc: float,  # Location of interface
                              nanop_shift: np.ndarray,  # Shift of np from ave
                              nanop_center: np.ndarray
                              ) -> None:
        """
        Define the annulus sections around the nanoparticles and then
        count the number of ODN in each annulus.
        """
        from scipy.signal import savgol_filter  # Import the savgol_filter
        counts, radii_distance = self.count_odn_in_annuluses(res_arr, nanop_center, nanoparticle_radius=20, dr=5)
        
        # Create subplots for each frame
        den_fig, den_ax = plt.subplots()
        average_counts = np.average(counts[100:], axis=0)
        den_ax.plot(average_counts, label=f'Frame')
        # for frame in range(100, self.nr_dict['nr_frames'], 10):
            # non_zero_indices = np.nonzero(counts[frame])  # Get indices of non-zero counts
            # if len(non_zero_indices[0]) > 0:
                # max_window_length = len(non_zero_indices[0]) // 2
                # smoothed_counts = savgol_filter(counts[frame][non_zero_indices], window_length=max_window_length, polyorder=5)  # Apply Savitzky-Golay smoothing
                # print(counts[frame][non_zero_indices])
                # den_ax.plot(non_zero_indices[0], smoothed_counts, label=f'Frame {frame}')
        den_ax.set_xlabel('Annulus Index')
        den_ax.set_ylabel('ODN Count')
        den_ax.set_title('ODN Counts in Annuluses')
        # den_ax.legend()
        plt.show()




    def count_odn_in_annuluses(self, odn_data, nanoparticle_center, nanoparticle_radius, dr):
        center_x, center_y, center_z = nanoparticle_center
        radii = np.arange(nanoparticle_radius, self.box_dims['x_hi'], dr)  # Create radii array with steps of dr
        radii_distance = np.linspace(nanoparticle_radius, self.box_dims['x_hi'], len(radii))
        num_time_frames, num_columns = odn_data.shape
        num_residues = num_columns // 3

        counts = np.zeros((num_time_frames, len(radii) - 1), dtype=int)

        for frame in range(num_time_frames):
            for residue in range(num_residues):
                x = odn_data[frame, residue*3]
                y = odn_data[frame, residue*3 + 1]
                z = odn_data[frame, residue*3 + 2]

                distance = self.calculate_distance(x, y, z, center_x, center_y, center_z)
                # print(distance)
                # Ensure the distance is within nanoparticle_radius
                if distance >= nanoparticle_radius:
                    annulus_idx = self.categorize_into_annuluses(nanoparticle_radius, distance, radii, dr) - 1
                    counts[frame, annulus_idx] += 1

        return counts, radii_distance

    def calculate_distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def categorize_into_annuluses(self, R, distances, radii, dr):
        annuluses = np.digitize(distances, radii)
        # annuluses = np.minimum(annuluses, int(np.ceil(R / dr)))  # Ensure annulus index doesn't exceed number of annuluses
        return annuluses


    def plot_odn(self,
                 res_arr: np.ndarray,  # ODN array
                 interface_loc: float,  # Location of the interface
                 nanop_shift: np.ndarray,  # Shift of the nanoparticle from ave
                 nanop_center: np.ndarray
                 ) -> None:
        """Plot ODN center of the masses"""
        odn_fig, odn_ax = plt.subplots()
        axis_names = ['x', 'y', 'z']
        odn_data: dict[str, np.ndarray] = {}
        for axis_idx, axis in enumerate(axis_names):
            odn_data[axis] = res_arr[:, axis_idx::3]
        # Find the ODN at the interface
        mask = (odn_data['z'] < interface_loc+10) & \
               (odn_data['z'] > interface_loc-10) 
        # Get the indices where the mask is True for each row
        indices: list[np.ndarray] = \
            [np.where(row_mask)[0] for row_mask in mask]
        for i_step in range(self.nr_dict['nr_frames']):
            x_data = \
                odn_data['x'][i_step][indices[i_step]] - nanop_shift[i_step][0]
            y_data = \
                odn_data['y'][i_step][indices[i_step]] - nanop_shift[i_step][1]
            odn_ax.scatter(x_data,
                           y_data,
                           s=5,
                           c='black',
                           alpha=(i_step+1)/self.nr_dict['nr_frames'])
        circ: matplotlib.patches.Circle = \
            self.__mk_circle(center=(nanop_center[0], nanop_center[1]))
        odn_ax.add_artist(circ)
        plt.axis('equal')
        odn_fig.savefig('test_odn.png')

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
    PlotOdnAnalysis()

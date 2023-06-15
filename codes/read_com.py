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
"""


import pickle
import numpy as np
import static_info as stinfo
import matplotlib.pylab as plt
import matplotlib


class ReadCom:
    """reading the center of mass file, the name is set static_info.py
    """
    def __init__(self) -> None:
        self.f_name: str = stinfo.files['com_pickle']
        self.com_arr: np.ndarray = self.get_data()
        self.box_dims: dict[str, float]  # Box dimensions, from stinfo
        self.box_dims = self.__get_box_dims()
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
            number_frame: int = 2
            for i in range(number_frame):
                if res in ['ODN', 'CLA']:
                    x_data, y_data, _ = \
                        self.__get_interface_oda(res_arr[i, x_indices],
                                                 res_arr[i, y_indices],
                                                 res_arr[i, z_indices],
                                                 res)
                    ax_com.scatter(x_data, y_data, s=5, c='black',
                                   alpha=(i+1)/number_frame)
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
        self.__plot_interface_z(interface_locz)

    def get_data(self) -> np.ndarray:
        """reading the file"""
        with open(self.f_name, 'rb') as f_rb:
            com_arr = pickle.load(f_rb)
        return com_arr

    @staticmethod
    def __plot_interface_z(interface_locz: list[tuple[float, float]]  # Z data
                           ) -> None:
        """save the plot for the water interface"""
        # Get the main data to plot
        z_average: list[float] = [item[0] for item in interface_locz]
        z_std_err: list[float] = [item[1] for item in interface_locz]
        # The main figure
        fig_z, ax_z = plt.subplots()
        # The x axis which is the number of frames
        x_range: range = range(len(interface_locz))
        # Plot the main data
        ax_z.plot(x_range, z_average, label='average z')
        # Calculate the upper and lower bounds
        upper_bound = [ave + err for ave, err in zip(z_average, z_std_err)]
        lower_bound = [ave - err for ave, err in zip(z_average, z_std_err)]
        # Geting the extermum values
        x_hi: np.int64 = np.max(x_range)  # Bounds of the x_range
        x_lo: np.int64 = np.min(x_range)  # Bounds of the x_range
        z_hi: float = stinfo.box['z'] / 2  # For the main plot
        z_lo: float = -z_hi   # For the main plot
        # Fill between the bounds
        ax_z.fill_between(x_range,
                          upper_bound,
                          lower_bound,
                          color='lightblue',
                          alpha=0.5,
                          label='Std Err')
        plt.axhline(y=0, color='red', linestyle='--', label='COM of NP')
        # Add labels and legend for the main plot
        ax_z.set_ylim([z_lo, z_hi])
        ax_z.set_xlim([x_lo, x_hi])
        ax_z.set_xlabel('time frame [index]')
        ax_z.set_ylabel('z [A]')
        ax_z.legend()

        # Create an inset plot
        # Position and size of the inset plot
        left, bottom, width, height = [0.2, 0.2, 0.66, 0.27]
        inset_ax = plt.axes([left, bottom, width, height])
        z_hi = np.floor(np.max(upper_bound)) + 1  # For the inset plot
        z_lo = np.floor(np.min(lower_bound)) - 1  # For the inset plot
        # Plot the inset curve
        inset_ax.plot(x_range, z_average)
        inset_ax.fill_between(x_range,
                              upper_bound,
                              lower_bound,
                              color='lightblue',
                              alpha=0.5)
        # x_lim_inset = np.floor(np.max(x_range))
        # Set the limits for the inset plot
        # inset_ax.set_xlim([1, 8])
        inset_ax.set_xlim([x_lo, x_hi])
        inset_ax.set_ylim([z_lo, z_hi])
        # Add labels and legend for the inset plot
        plt.savefig('test.png')
        plt.close(fig_z)

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
    def __get_box_dims() -> dict[str, float]:
        """return the box lims for plotting."""
        box_dims: dict[str, float] = {}  # Box dimensions
        if stinfo.box['centered']:
            box_dims['x_hi'] = np.ceil(stinfo.box['x']/2)
            box_dims['x_lo'] = -np.ceil(stinfo.box['x']/2)
            box_dims['y_hi'] = np.ceil(stinfo.box['y']/2)
            box_dims['y_lo'] = -np.ceil(stinfo.box['y']/2)
            box_dims['z_hi'] = np.ceil(stinfo.box['z']/2)
            box_dims['z_lo'] = -np.ceil(stinfo.box['z']/2)
        else:
            box_dims['x_hi'] = np.ceil(stinfo.box['x'])
            box_dims['x_lo'] = 0.0
            box_dims['y_hi'] = np.ceil(stinfo.box['y'])
            box_dims['y_lo'] = 0.0
            box_dims['z_hi'] = np.ceil(stinfo.box['z'])
            box_dims['z_lo'] = 0.0
        return box_dims

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
        column_x_indices = np.where(self.com_arr[-1] ==
                                    stinfo.reidues_id[res])
        res_arr: np.ndarray = np.squeeze(self.com_arr[:, column_x_indices])
        return res_arr

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
        x_indices = range(3, res_arr.shape[1], 3)
        y_indices = range(4, res_arr.shape[1], 3)
        z_indices = range(5, res_arr.shape[1], 3)
        return x_indices, y_indices, z_indices

    @staticmethod
    def __get_interface_oda(x_data: np.ndarray,  # All the x values for the oda
                            y_data: np.ndarray,  # All the x values for the oda
                            z_data: np.ndarray,  # All the x values for the oda
                            res: str  # Name of the residue
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get xyz for all the oda at the interface and or ion in the bolck"""
        interface_z: float = 20  # should be calculated, interface treshhold
        if res == 'ODN':
            inface_indx = np.where(z_data > interface_z)[0]  # At interface
        elif res == 'CLA':
            inface_indx = np.where(z_data < interface_z)[0]  # At interface
        return x_data[inface_indx], y_data[inface_indx], z_data[inface_indx]

    @staticmethod
    def __mk_circle() -> matplotlib.patches.Circle:
        radius = stinfo.np_info['radius']
        circle = plt.Circle((0, 0),
                            radius,
                            color='red',
                            linestyle='dashed',
                            fill=False, alpha=1)
        return circle


if __name__ == '__main__':
    data = ReadCom()

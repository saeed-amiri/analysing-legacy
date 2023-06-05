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
        self.com_arr: np.ndarray = self.__get_data()
        self.__plot_com()

    def __get_data(self) -> np.ndarray:
        """reading the file"""
        with open(self.f_name, 'rb') as f_rb:
            com_arr = pickle.load(f_rb)
        return com_arr

    def __plot_com(self) -> None:
        """test plotting"""
        ax_com = plt.gca()
        x_indices: range  # Range of the indices
        y_indices: range  # Range of the indices
        z_indices: range  # Range of the indices
        for res in ['ODN', 'CLA', 'SOL']:
            res_arr: np.ndarray = self.__get_residue(res)
            x_indices, y_indices, z_indices = self.__get_xy_com(res_arr)
            number_frame: int = 11
            for i in range(number_frame):
                if res in ['ODN', 'CLA']:
                    x_data, y_data, z_data = \
                        self.__get_interface_oda(res_arr[i, x_indices],
                                                 res_arr[i, y_indices],
                                                 res_arr[i, z_indices],
                                                 res)
                    ax_com.scatter(x_data, y_data, s=5, c='black',
                                   alpha=(i+1)/number_frame)
            self.__plot_odn_com(ax_com, res)

    def __get_residue(self,
                      res: str,  # Name of the residue to get the data,
                      ) -> np.ndarray:
        """return the info for the selected residue"""
        column_x_indices = np.where(self.com_arr[-1] ==
                                    stinfo.reidues_id[res])
        res_arr: np.ndarray = np.squeeze(self.com_arr[:, column_x_indices])
        return res_arr

    @staticmethod
    def __plot_odn_com(ax_com: matplotlib.axes,  # The center of mass data
                       res: str  # Name of the residue to save file
                       ) -> None:
        """plot and save the COM of the ODA"""
        # Create a circle with origin at (0, 0) and radius of the nanoparticle
        r_np = stinfo.np_info['radius']
        circle = plt.Circle((0, 0), r_np, color='red', fill='True', alpha=0.25)

        # Get the current axes and add the circle to the plot
        ax_com.add_artist(circle)
        ax_com.set_aspect('equal')
        ax_com.set_xlim(-109, 109)
        ax_com.set_ylim(-109, 109)

        ax_com.set_xlabel("x component [A]")
        ax_com.set_ylabel("y component [A]")
        plt.title(f'center of mass of {res} around NP')
        pname: str  # Name of the output file
        pname = f'{res}_com.png'
        plt.savefig(pname, bbox_inches='tight', transparent=False)

    @staticmethod
    def __get_xy_com(res_arr: np.ndarray,  # All the times
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
        """get xyz for all the oda at the interface"""
        interface_z: float = 10  # should be calculated, interface treshhold
        if res == 'ODN':
            inface_indx = np.where(z_data > interface_z)[0]  # At interface
        elif res == 'CLA':
            inface_indx = np.where(z_data < interface_z)[0]  # At interface
        return x_data[inface_indx], y_data[inface_indx], z_data[inface_indx]


if __name__ == '__main__':
    data = ReadCom()

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
        for mol in ['ODN']:
            column_x_indices = np.where(self.com_arr[-1] ==
                                        stinfo.reidues_id[mol])
            oda: np.ndarray = np.squeeze(self.com_arr[:, column_x_indices])
            x_indices = range(3, oda.shape[1], 3)  # Indices on the x-axis
            y_indices = range(4, oda.shape[1], 3)  # Indices on the y-axis
            for i in range(12):
                x_data = oda[i, x_indices]
                y_data = oda[i, y_indices]
                plt.scatter(x_data, y_data)
        # Set the aspect ratio to 'equal'
        plt.gca().set_aspect('equal')
        plt.show()


if __name__ == '__main__':
    data = ReadCom()

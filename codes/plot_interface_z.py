"""plot water surface at inerface"""

import numpy as np
import matplotlib.pylab as plt
import static_info as stinfo


class PlotInterfaceZ:
    """plot the water surface at the interface.
    The protonation of APTES is affected by changes in the water level.
    To determine this, we calculate the average z component of all the
    center of mass of the water molecules at the interface.
    """
    def __init__(self,
                 locz: list[tuple[float, float]]  # Z and standard diviation
                 ) -> None:
        self.upper_bound: list[float]  # Upper bound of the err bar
        self.lower_bound: list[float]  # Lower bound of the err bar
        self.z_average: list[float]  # Average value of the z component
        self.z_std_err: list[float]  # Error (std or std_err) of the z
        self.x_range: range  # Range of data on the x axis
        self.z_average, self.z_std_err, self.x_range = self.__get_data(locz)
        self.plot_interface_z()

    def plot_interface_z(self):
        """call the functions"""
        self.__get_bounds()
        self.__mk_main_graph()

    def __mk_main_graph(self) -> None:
        """plot the main graph"""
        fig_i: plt.figure  # Canvas
        ax_i: plt.axes  # main axis
        fig_i, ax_i = self.__mk_canvas()
        ax_i.errorbar(self.x_range,
                      self.z_average,
                      yerr=self.z_std_err,
                      fmt='o',
                      markersize=2,
                      capsize=4,
                      label='std')
        std_max: np.float64 = np.max(np.abs(self.z_std_err))
        ax_i.set_ylim([np.min(self.z_average)-std_max,
                       np.max(self.z_average)+std_max])
        plt.legend(loc='upper right')
        fig_i.savefig('main_graph.png')
        plt.close(fig_i)

    def __mk_canvas(self) -> tuple[plt.figure, plt.axes]:
        """make the pallete for the figure"""
        fig_main, ax_main = plt.subplots()
        x_hi: np.int64  # Bounds of the self.x_range
        x_lo: np.int64  # Bounds of the self.x_range
        z_hi: float  # For the main plot
        z_lo: float  # For the main plot
        x_hi, x_lo, z_hi, z_lo = self.__get_lims()
        ax_main.set_xlabel('frame index')
        ax_main.set_ylabel('z [A]')
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(z_lo, z_hi)
        return fig_main, ax_main

    def __get_bounds(self) -> None:
        """calculate the bunds of error bar for surface"""
        self.upper_bound = [ave + err for ave, err in
                            zip(self.z_average, self.z_std_err)]
        self.lower_bound = [ave - err for ave, err in
                            zip(self.z_average, self.z_std_err)]

    def __get_lims(self) -> tuple[np.int64, np.int64, float, float]:
        """get the limits for the axis"""
        # Geting the extermum values
        x_hi: np.int64 = np.max(self.x_range)  # Bounds of the self.x_range
        x_lo: np.int64 = np.min(self.x_range)  # Bounds of the self.x_range
        z_hi: float = stinfo.box['z'] / 2  # For the main plot
        z_lo: float = -z_hi   # For the main plot
        return x_hi, x_lo, z_hi, z_lo

    @staticmethod
    def __get_data(locz: list[tuple[float, float]]  # Z and standard diviation
                   ) -> tuple[list[float], list[float], range]:
        """get tha data"""
        z_average: list[float] = [item[0] for item in locz]
        z_std_err: list[float] = [item[1] for item in locz]
        x_range: range = range(len(locz))
        return z_average, z_std_err, x_range

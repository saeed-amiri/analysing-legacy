"""plot water surface at inerface"""

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import static_info as stinfo
import plot_tools as ptools


class PlotInterfaceZ:
    """plot the water surface at the interface.
    The protonation of APTES is affected by changes in the water level.
    To determine this, we calculate the average z component of all the
    center of mass of the water molecules at the interface.
    """
    fontsize: int = 14  # Fontsize for all in plots
    transparent: bool = False  # Save fig background
    f_errbar: str = 'errbar.png'  # Name of the errbar figure
    f_inset: str = 'inset.png'  # Name of the inset figure

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
        # Get the bonds for the plotting
        self.__get_bounds()
        # Plot the graph with errobar
        self.__mk_errorbar()
        # Plot the graph with inset
        self.__mk_inset()
        # Plot the graph with shadow area
        self.__mk_shadow_area()

    def __mk_errorbar(self) -> None:
        """plot the main graph"""
        fig_i: plt.figure  # Canvas
        ax_i: plt.axes  # main axis
        fig_i, ax_i = self.__mk_canvas()
        ax_i.errorbar(self.x_range,
                      self.z_average,
                      yerr=self.z_std_err,
                      color='k',
                      fmt='o',
                      markersize=4,
                      markeredgecolor='k',
                      markerfacecolor='k',
                      capsize=4,
                      linewidth=1,
                      zorder=1,
                      label='data and std')
        std_max: np.float64 = np.max(np.abs(self.z_std_err))
        ax_i.set_ylim([np.min(self.z_average)-2*std_max,
                       np.max(self.z_average)+2*std_max])
        ax_i = self.__plot_ave(ax_i)
        self.save_fig(fig_i, ax_i, fname=self.f_errbar)

    def __mk_inset(self) -> None:
        fig_i: plt.figure  # Canvas
        ax_i: plt.axes  # main axis
        fig_i, ax_i = self.__mk_canvas()
        self.__plot_inset()
        ax_i = self.__plt_data_and_fill(ax_i)
        ax_i = self.__plot_ave(ax_i)
        self.save_fig(fig_i, ax_i, fname=self.f_inset)

    def __plot_inset(self) -> plt.axes:
        """Create an inset plot"""
        # Position and size of the inset plot
        left, bottom, width, height = [0.2, 0.2, 0.66, 0.27]
        inset_ax = plt.axes([left, bottom, width, height])
        # Plot the inset curve
        inset_ax = self.__plt_data_and_fill(inset_ax)
        inset_ax = self.__plot_ave(inset_ax)
        # Set the limits for the inset plot
        x_hi = np.floor(np.max(self.x_range))
        x_lo = np.floor(np.min(self.x_range))
        z_hi = np.floor(np.max(self.upper_bound)) + 1
        z_lo = np.floor(np.min(self.lower_bound)) - 1
        inset_ax.set_xlim([x_lo, x_hi])
        inset_ax.set_ylim([z_lo, z_hi])
        inset_ax = self.__set_ax_font_label(inset_ax, fsize=11)
        return inset_ax

    def __plt_data_and_fill(self,
                            ax_i: plt.axes  # The ax to plot on
                            ) -> plt.axes:
        """plot the graph and fill in std"""
        ax_i.plot(self.x_range, self.z_average, color='k', label='average z')
        ax_i.fill_between(self.x_range,
                          self.upper_bound,
                          self.lower_bound,
                          color='k',
                          label='std',
                          alpha=0.5)
        return ax_i

    def __mk_shadow_area(self) -> None:
        """plot a graph showing the area for water and oil as a shadow"""
        fig_i: plt.figure  # Canvas
        ax_i: plt.axes  # main axis
        fig_i, ax_i = self.__mk_canvas()
        self.__plt_data_and_fill(ax_i)
        # Get the limits of the x-axis and y-axis
        y_min, y_max = ax_i.get_ylim()
        ax_i.fill_between(self.x_range,
                          self.lower_bound,
                          y_min,
                          color='royalblue',
                          alpha=0.5,
                          edgecolor='none')
        ax_i.fill_between(self.x_range,
                          self.upper_bound,
                          y_max,
                          color='yellow',
                          alpha=0.5,
                          edgecolor='none')
        self.save_fig(fig_i, ax_i, 'test.png')

    def __plot_ave(self,
                   ax_i: plt.axes,  # axes to plot average
                   lw: int = 1  # Width of the line
                   ) -> plt.axes:
        """plot average of z_average"""
        z_mean: np.float64 = np.mean(self.z_average)
        ax_i.axhline(z_mean, color='r', ls='--', lw=lw,
                     label=f'mean of data: {z_mean:.2f}', zorder=2)
        return ax_i

    def __mk_canvas(self) -> tuple[plt.figure, plt.axes]:
        """make the pallete for the figure"""
        width = stinfo.plot['width']
        fig_main, ax_main = plt.subplots(1, figsize=ptools.set_sizes(width))
        # Set font for all elements in the plot
        x_hi: np.int64  # Bounds of the self.x_range
        x_lo: np.int64  # Bounds of the self.x_range
        z_hi: float  # For the main plot
        z_lo: float  # For the main plot
        x_hi, x_lo, z_hi, z_lo = self.__get_lims()
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(z_lo, z_hi)
        num_xticks = 5
        xticks = np.linspace(self.x_range[0], self.x_range[-1], num_xticks)
        ax_main.set_xticks(xticks)
        ax_main = self.__set_ticks(ax_main)
        ax_main = self.__set_ax_font_label(ax_main)
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

    @classmethod
    def __set_ax_font_label(cls,
                            ax_main: plt.axes,  # Main axis to set parameters
                            fsize: int = 0  # font size if called with font
                            ) -> plt.axes:
        """set parameters on the plot"""
        if fsize == 0:
            fontsize = cls.fontsize
            ax_main.set_xlabel('frame index', fontsize=fontsize)
            ax_main.set_ylabel('z [A]', fontsize=fontsize)
        else:
            fontsize = fsize
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.size'] = fontsize
        ax_main.tick_params(axis='x', labelsize=fontsize)
        ax_main.tick_params(axis='y', labelsize=fontsize)
        return ax_main

    @classmethod
    def save_fig(cls,
                 fig: plt.figure,  # The figure to save,
                 axs: plt.axes,  # Axes to plot
                 fname: str,  # Name of the output for the fig
                 loc: str = 'upper right'  # Location of the legend
                 ) -> None:
        """to save all the fige"""
        legend = axs.legend(loc=loc, bbox_to_anchor=(1.0, 1.0))
        legend.set_bbox_to_anchor((1.0, 1.0))
        fig.savefig(fname,
                    dpi=300,
                    pad_inches=0.1,
                    edgecolor='auto',
                    bbox_inches='tight',
                    transparent=cls.transparent
                    )
        plt.close(fig)

    @staticmethod
    def __set_ticks(ax_main: plt.axes  # The axes to wrok with
                    ) -> plt.axes:
        """set tickes"""
        ax2 = ax_main.twiny()
        ax3 = ax_main.twinx()
        ax_main.tick_params(axis='both', direction='in')
        ax2.set_xticklabels([])  # Remove the tick labels on the top x-axis
        ax2.tick_params(axis='x', direction='in')
        ax3.set_yticklabels([])  # Remove the tick labels on the top x-axis
        ax3.tick_params(axis='y', direction='in')
        for ax_i in [ax_main, ax2]:
            ax_i.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            ax_i.xaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(n=5))
            ax_i.tick_params(which='minor', direction='in')
        for ax_i in [ax_main, ax3]:
            ax_i.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            ax_i.yaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(n=5))
            ax_i.tick_params(which='minor', direction='in')
        return ax_main

    @staticmethod
    def __get_data(locz: list[tuple[float, float]]  # Z and standard diviation
                   ) -> tuple[list[float], list[float], range]:
        """get tha data"""
        z_average: list[float] = [item[0] for item in locz]
        z_std_err: list[float] = [item[1] for item in locz]
        x_range: range = range(len(locz))
        return z_average, z_std_err, x_range


def generate_float_tuples(list_len: int  # length of the list
                          ) -> list[tuple[float, float]]:
    """
    Generate a list of tuples with float values.
    for running the scripts alone.

    Args:
        n (int): The number of tuples to generate.

    Returns:
        list[tuple[float, float]]: The list of tuples containing float values.
    """
    float_tuples: list[tuple[float, float]] = []
    for _ in range(list_len):
        float_tuples.append((np.random.rand(), 10*np.random.rand()))
    return float_tuples


if __name__ == '__main__':
    PlotInterfaceZ(generate_float_tuples(51))

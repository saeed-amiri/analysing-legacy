"""plot tesnsion for different system
In Gromacs the surface tension unit is (bar.nm).
########### Surface Tension Unit Conversion at T=298k ##########
#( bar.nm ) / 41.14 >>> k_BT/nm^2
#(k_BT/nm^2) * 4.113 >>> mN/m
#( bar.nm ) / 10 >>> mN/m
################################################################
"""


import sys
import typing
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pylab as plt

import plot_tools
import static_info as stinfo
from colors_text import TextColor as bcolors


def set_sizes(width_pic, fraction=1):
    """set figure dimennsion"""
    fig_width_pt = width_pic*fraction
    inches_per_pt = 1/72.27
    golden_ratio = (5**0.5 - 1)/2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


# Update the rcParams with the desired configuration settings
mpl.rcParams['axes.prop_cycle'] = \
    plt.cycler('color', ['k', 'r', 'b', 'g']) + \
    plt.cycler('ls', ['-', '--', ':', '-.'])
mpl.rcParams['figure.figsize'] = (3.3, 2.5)
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'


# Set axis style
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tick_params(direction='in')


class GetLog:
    """read log file contains tesnsion"""

    tesnsion_unit: str = 'mN/m2'
    tesnsion_ratio: float = 2 * 10
    interface_are: float = 21.7 * 21.7

    def __init__(self,
                 filename: str
                 ) -> None:
        self.filename: str = filename
        raw_tesnsion: pd.DataFrame = self.read_data()
        print(self.proccess_tensions(raw_tesnsion))

    def read_data(self) -> pd.DataFrame:
        """check and read lines and data"""
        data: list[dict[str, typing.Union[str, float]]] = []
        try:
            with open(self.filename, 'r', encoding='utf8') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) == 4:
                            name = parts[0]
                            oda_nr = int(parts[1])
                            no_nanop = float(parts[2])
                            with_nanop = float(parts[3])
                            data.append({"Name": name,
                                         "Oda": oda_nr,
                                         "NoWP": no_nanop,
                                         "WP": with_nanop})
                        else:
                            print(f"Skipping line: {line}")
            return pd.DataFrame(data)
        except FileNotFoundError:
            sys.exit(f"File '{self.filename}' not found.")

    def proccess_tensions(self,
                          raw_tension: pd.DataFrame
                          ) -> pd.DataFrame:
        """convert the tension and add them as extera columns"""
        df_c: pd.DataFrame = raw_tension.copy()
        df_c['sigma'] = raw_tension['Oda'] / self.interface_are
        df_c['NoWP_mN'] = raw_tension['NoWP'] / self.tesnsion_ratio
        df_c['WP_mN'] = raw_tension['WP'] / self.tesnsion_ratio
        df_c['delta_NoWP'] = df_c['NoWP_mN'] - df_c['NoWP_mN'][0]
        df_c['delta_WP'] = df_c['WP_mN'] - df_c['NoWP_mN'][0]
        return df_c


if __name__ == "__main__":
    data_processor = GetLog(sys.argv[1])

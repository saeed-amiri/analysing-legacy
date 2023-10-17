"""read and plot COM of NP calculated by plumed"""

import re
import sys
import typing
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt

import plot_tools


class GetData:
    """read file contains tesnsion"""

    def __init__(self,
                 filename: str
                 ) -> None:
        self.filename: str = filename
        com_df: pd.DataFrame = self.read_data()
        self.com_df = self.get_zero_point(com_df, 10)

    def get_zero_point(self,
                       com_df: pd.DataFrame,
                       nr_ave_point: int = 10
                       ) -> pd.DataFrame:
        """remove the average of the first points to brings data
        zero"""
        df_c: pd.DataFrame = com_df.copy()
        
        df_c['x_center'] = com_df['x'] - np.average(com_df['x'][:nr_ave_point])
        df_c['y_center'] = com_df['y'] - np.average(com_df['y'][:nr_ave_point])
        df_c['z_center'] = com_df['z'] - np.average(com_df['z'][:nr_ave_point])
        return df_c

    def read_data(self) -> pd.DataFrame:
        """check and read lines and data"""
        data: list[dict[str, typing.Union[str, float]]] = []
        try:
            with open(self.filename, 'r', encoding='utf8') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) == 4 and not line.startswith('#'):
                            time = int(float(parts[0]))
                            oda_nr = float(parts[1])
                            no_nanop = float(parts[2])
                            with_nanop = float(parts[3])
                            data.append({"time": time,
                                         "x": oda_nr,
                                         "y": no_nanop,
                                         "z": with_nanop})
            return pd.DataFrame(data)
        except FileNotFoundError:
            sys.exit(f"File '{self.filename}' not found.")


class PlotCom:
    """plot com"""

    selected_oda: list[int] = [5]
    # selected_oda: list[int] = [5, 15, 20, 200]

    def __init__(self,
                 filenames: list[str]
                 ) -> None:
        self.files: list[str] = self.order_files(filenames)
        df_x: pd.DataFrame
        df_y: pd.DataFrame
        df_z: pd.DataFrame
        df_x, df_y, df_z = self.mk_xyzdfs()
        self.initiate_plots(df_x, df_y, df_z)

    def order_files(self,
                    filenames: list[str]
                    ) -> None:
        """sort filenames"""
        return sorted(filenames, key=self.__extract_numeric_part)

    def mk_xyzdfs(self) -> tuple[pd.DataFrame, ...]:
        """make dataframes for x and y and z"""
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        df_z = pd.DataFrame()
        for com_f in self.files:
            match = re.search(r'\d+', com_f)
            column = match.group()
            df_i: pd.DataFrame = GetData(com_f).com_df
            df_x[column] = df_i['x_center']
            df_y[column] = df_i['y_center']
            df_z[column] = df_i['z_center']
        return df_x, df_y, df_z
    
    @staticmethod
    def __extract_numeric_part(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0

    def initiate_plots(self,
                       df_x: pd.DataFrame,
                       df_y: pd.DataFrame,
                       df_z: pd.DataFrame
                       ) -> None:
        """set canvas and plot com in different fashions"""
        df_axes = [df_x, df_y, df_z]
        df_names = ['x', 'y', 'z']
        for df_i, df_name in zip(df_axes, df_names):
            fig_i, ax_i = plot_tools.mk_canvas((0, 200),
                                               num_xticks=6,
                                               fsize=12,
                                               add_xtwin=False)
            self._plot_df_i(df_i, ax_i)
            ax_i.set_ylabel(f'{df_name} [nm]')
            plot_tools.save_close_fig(
                fig=fig_i, axs=ax_i, fname=f'com{df_name}.png')
        self.plot_xyz()

    def plot_xyz(self) -> None:
        """plot xyz in one graph"""
        for oda in self.selected_oda:
            com_f: str = f'npCom{oda}.dat'
            df_i: pd.DataFrame = GetData(com_f).com_df
            fig_i, ax_i = plot_tools.mk_canvas((0, 200),
                                               num_xticks=6,
                                               fsize=12,
                                               add_xtwin=False)
            ax_i.set_ylabel(f'COM [nm]')
            ax_i.plot(df_i['x_center'], label='x')
            ax_i.plot(df_i['y_center'], label='y')
            ax_i.plot(df_i['z_center'], label='z')
            plot_tools.save_close_fig(
                fig=fig_i, axs=ax_i, fname=f'com{oda}.png')

    def _plot_df_i(self,
                   df_i: pd.DataFrame,
                   ax_i: plt.axes
                   ) -> plt.axes:
        colors: list[str] = ['r', 'b', 'k', 'g']
        styles: list[str] = [':', '--', '-', ':', '-', '--', '-.', ':', '-']
        for i, col in enumerate(self.selected_oda):
            ax_i.plot(df_i[str(col)][1:],
                      ls=styles[i],
                      c=colors[i],
                      label=f'WP{col}',
                      )
        return ax_i


if __name__ == "__main__":
    PlotCom(sys.argv[1:])

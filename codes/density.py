"""plot energy output of GROMACS from command line:
    `gmx_mpi density -s npt.tpr -f npt.trr -n index.ndx'
    Input:
    Any number of input file with xvg extensions
    """

import re
import sys
import typing
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
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


class GetBulkDensity:
    """find the density of the D10 and SOL
    In this class, we scan through a list of xvg files and identify
    the locations where all of them have a value of zero.
    """

    residues: list[str] = ['APT', 'COR', 'ODN']  # Name of all the residues
    water_res: str = 'SOL'  # Water residue
    oil_res: str = 'D10'  # Oil residue
    ion_res: str = 'CLA'  # Ion residue
    other_res: list[str] = ['COR_APT']
    info_msg: str = 'Message:\n'  # To add messages

    water_bulk: np.float64  # Density of the bulk water
    oil_bulk: np.float64  # Density of the bulk oil
    residues_data: dict[str, typing.Any]   # All data read for residues

    def __init__(self) -> None:
        self.get_bulk()
        print(f'{bcolors.OKBLUE}{self.__class__.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')

    def get_bulk(self) -> None:
        """get the bulk density"""
        residues_data: dict[str, typing.Any] = self.read_residues()
        self.water_bulk = \
            self.get_bulk_density(residues_data, self.water_res, self.oil_res)
        self.oil_bulk = \
            self.get_bulk_density(residues_data, self.oil_res, self.water_res)
        self.residues_data = residues_data

    def get_bulk_density(self,
                         residues_data: dict[str, typing.Any],  # All the data
                         target: str,  # Wanted bulk density
                         extension: str  # The other of the targets
                         ) -> np.float64:
        """get bulk of water an oil"""
        indices: list[int] = self.find_bulk(residues_data, extension)
        bulk_area = residues_data[target]['data'][indices][:, 1]
        bulk_density: np.float64 = \
            np.average(bulk_area)
        self.info_msg += f'\tThe bulk density for {target} is average of:\n'
        self.info_msg += f'\t{bulk_area}\n'
        return bulk_density

    def find_bulk(self,
                  residues_data: dict[str, typing.Any],  # All the info for res
                  extension: str  # Water or oil
                  ) -> list[int]:
        """find bulk density of water and oil"""
        res_list: list[str] = self.residues.copy()
        res_list.extend([extension])
        arr_list: list[np.ndarray]  # density of all the data
        arr_list = [residues_data[item]['data'] for item in res_list]
        indices: list[int]  # Where others are zero
        # Find the indices where all arrays have zero in their second column
        indices = np.where(np.all(
            np.array([arr[:, 1] <= 0.1 for arr in arr_list]), axis=0))[0]
        return indices

    def read_residues(self) -> dict[str, typing.Any]:
        """read every residues in the list"""
        all_reses: list[str] = self.residues.copy()
        all_reses.extend([self.water_res, self.oil_res, self.ion_res])
        all_reses.extend(self.other_res)
        residues_data: dict[str, typing.Any] = {}  # Density of all
        for xvg_f in all_reses:
            fname = f'{xvg_f}.xvg'
            data = self.__get_header(fname)
            residues_data[xvg_f] = data
            residues_data[xvg_f]['data'] = self.__read_data(data['data'])
        return residues_data

    @staticmethod
    def __read_data(data: list  # Unbrocken lines of data
                    ) -> np.ndarray:
        data_arr: np.ndarray  # Array to save data
        data_arr = np.zeros((len(data), 2))
        for ind, item in enumerate(data):
            tmp_list = [i for i in item.split(' ') if i]
            data_arr[ind][0] = float(tmp_list[0])
            data_arr[ind][1] = float(tmp_list[1])
            del tmp_list
        return data_arr

    def __get_header(self,
                     fname: str  # Name of the xvg 25
                     ) -> dict[str, typing.Any]:
        """read header of xvg file, lines which started with `@`"""
        self.info_msg += f'\tReading file : {fname}\n'
        max_lines: int = 30  # should not be more then 26
        linecount: int = 0  # Tracking number of lines with `@`
        header_line: int = 0  # Number of header lines
        data_dict: dict = dict()  # Contains all the labels from xvg file
        data: list = []  # Lines contains numberic data
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                linecount += 1
                line: str = f_r.readline()
                if line.startswith('#') or line.startswith('@'):
                    header_line += 1
                    if line.strip().startswith('@'):
                        axis, label_i = self.__process_line(line.strip())
                        if axis:
                            if axis in ['xaxis', 'yaxis', 'legend']:
                                data_dict[axis] = label_i
                else:
                    if line:
                        data.append(line.strip())

                if linecount > max_lines and len(data_dict) < 3:
                    print(f'{bcolors.OKCYAN}{self.__class__.__name__}:\n'
                          f'\tNumber of lines with `@` in "{fname}" is '
                          f'{linecount}{bcolors.ENDC}')
                    break
                if not line:
                    break
        data_dict['nHeader'] = header_line
        data_dict['fname'] = fname
        data_dict['data'] = data
        return data_dict

    @staticmethod
    def __process_line(line: str  # Line start with `@`
                       ) -> tuple:
        """get labels from the lines"""
        l_line: list  # Breaking the line
        l_line = line.split('@')[1].split(' ')
        l_line = [item for item in l_line if item]
        axis: str = ''  # Label of the axis
        label_i: str = ''  # Label of the axis
        if 'label' in l_line:
            if 'xaxis' in l_line:
                axis = 'xaxis'
            if 'yaxis' in l_line:
                axis = 'yaxis'
            label_i = re.findall('"([^"]*)"', line)[0]
        elif 'legend' in l_line:
            if 's0' in l_line:
                label_i = re.findall('"([^"]*)"', line)[0]
                axis = 'legend'
        return axis, label_i


if __name__ == "__main__":
    res_data = GetBulkDensity()
    fig, ax = plt.subplots(1, figsize=set_sizes(width_pic=426.79135))
    xvg_files: list = sys.argv[1:]
    average: bool = True  # If write average on the legend
    transparent: bool = False  # If the figure wants to be transparent
    savefig_kwargs = {'fname': 'output.png',
                      'dpi': 300,
                      'bbox_inches': 'tight'
                      }
    if transparent:
        savefig_kwargs['transparent'] = 'true'
    for f in xvg_files:
        xvg = res_data.residues_data[f]
        label = f'{f}'
        bulk_value = np.float64(1.0)
        if f == 'SOL':
            bulk_value = res_data.water_bulk
            if average:
                label += f', bulk={bulk_value:.2f}'
        elif f == 'D10':
            bulk_value = res_data.oil_bulk
            if average:
                label += f', bulk={bulk_value:.2f}'

        ax.plot(xvg['data'][:, 0],
                xvg['data'][:, 1]/bulk_value,
                label=label)
    plt.xlabel(xvg['xaxis'])
    plt.ylabel(xvg['yaxis'])

    # Adjust tick label font size and style if needed
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend()
    plt.savefig(**savefig_kwargs)
    # plt.show()

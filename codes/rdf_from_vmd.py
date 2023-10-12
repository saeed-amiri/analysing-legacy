import sys
import matplotlib.pylab as plt

import plot_tools

class GetData:
    """
    read data
    """
    def __init__(self) -> None:
        self.filename: str = sys.argv[1]
        self.plot_data()

    def _parse_data(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        # Extract title
        title = \
            next(line.split(' ')[1] for line in lines if "@title" in line)

        # Extract x and y data
        data_lines = \
            [line for line in lines if not line.startswith("@") and
             line != '&\n']

        x_data, y_data = [], []
        for data in data_lines:
            x, y = data.split()
            x_data.append(float(x))
            y_data.append(float(y))

        return title, x_data, y_data

    def plot_data(self):
        title, x_data, y_data = self._parse_data()
        # Create subplots for each frame
        fig_i, ax_i = plot_tools.mk_canvas((0, 200),
                                           num_xticks=6,
                                           fsize=12)
        ax_i = plot_tools.set_ax_font_label(
            ax_i, fsize=12, x_label='Distance [nm]', y_label=title)
        ax_i.plot(x_data, y_data, c='k', label='WP10')
        plt.axhline(y=1, color='grey', linestyle='--', alpha=0.5)
        plt.title(title)
        plot_tools.save_close_fig(fig_i, ax_i, 'CLA_cdf')



if __name__ == '__main__':
    GetData()

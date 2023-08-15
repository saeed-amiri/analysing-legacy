"""tools for ploting"""

import matplotlib
import matplotlib.pylab as plt


def set_sizes(width: float,  # Width of the plot in points
              fraction: float = 1
              ) -> tuple[float, float]:
    """
    Calculate figure dimensions based on width and fraction.

    This function calculates the dimensions of a figure based on the
    desired width and an optional fraction. It uses the golden ratio
    to determine the height.

    Args:
        width (float): The width of the plot in points.
        fraction (float, optional): A fraction to adjust the width.
        Default is 1.

    Returns:
        tuple[float, float]: A tuple containing the calculated width
        and height in inches for the figure dimensions.
    """
    fig_width_pt = width*fraction
    inches_per_pt = 1/72.27
    golden_ratio = (5**0.5 - 1)/2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def mk_circle(radius: float,
              center: tuple[float, float] = (0, 0)
              ) -> matplotlib.patches.Circle:
    """
    Create a dashed circle.

    This function creates a dashed circle with the specified radius and
    center coordinates.

    Args:
        radius (float): The radius of the circle.
        center (tuple[float, float], optional): The center coordinates
        of the circle. Default is (0, 0).

    Returns:
        matplotlib.patches.Circle: A `Circle` object representing the
        dashed circle.
    """
    circle = plt.Circle(center,
                        radius,
                        color='red',
                        linestyle='dashed',
                        fill=False, alpha=1)
    return circle

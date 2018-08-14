from itertools import cycle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from google.colab import files
except:
    files = None


def tableu20_color_table():
    # These are the "Tableau 20" colors as RGB.
    def new_table():
        t = [
            (62, 139, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
            (188, 189, 34), (219, 219, 141), (93, 190, 207), (158, 218, 229)
        ]  # yapf:disable
        t = np.array(t) - 25
        t[t < 0] = 0
        return t

    tableau20 = new_table()

    color = tableau20[0, :]
    yield tuple(color / 255.0)

    distances = np.square(tableau20 - color).sum(axis=1)
    while True:
        next_color_index = distances.argmax()
        color = tableau20[next_color_index, :]
        # Scale the RGB values to the [0, 1] range, which is the format
        # matplotlib accepts.
        yield tuple(color / 255.0)

        tableau20 = np.delete(tableau20, [next_color_index], axis=0)
        if len(tableau20) < 1:
            tableau20 = new_table()
        distances = np.square(tableau20 - color).sum(axis=1)


def marker_table():
    while True:
        for k, v in sorted(
                mpl.markers.MarkerStyle.markers.items(),
                key=lambda x: (str(x[0]), str(x[1]))):
            if (
                v != "nothing"
                and v != "pixel"
                and v != "vline"
                and v != 'hline'
                #       v != 'star' and
                and v != 'tri_down'
                and v != 'tickleft'
                and v != 'tickup'
                and v != 'tri_up'
                and v != 'tickright'
                #       v != 'x' and
                and v != 'tickdown'
                # and v != 'tri_left'
                # and v != 'tri_right'
                and v != 'plus'
                and v != 'point'
            ):  # yapf:disable
                print(v)
                yield k


def line_style_table():
    return cycle(['-', '--', '-.', ':'])


def download_figure(file_name):
    plt.savefig(file_name)
    if files is not None:
        files.download(file_name)


def set_good_defaults(figure_style="seaborn-whitegrid"):
    plt.style.use(figure_style)
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.framealpha'] = 0.8
    mpl.rcParams['legend.frameon'] = True

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


def plot_sampled_functions(data,
                           x,
                           phi,
                           dist,
                           map_model,
                           num_samples,
                           mle_model=None,
                           color=None):
    colors = tableu20_color_table()
    plt.plot(data.good.phi, data.good.y, '.', markersize=15, color=color)
    plt.plot(
        data.bad.phi, data.bad.y, '.', markersize=15, color=color, alpha=0.2)

    line_styles = line_style_table()

    legends = ['observed (noise removed)', 'unobserved', 'MAP']

    plt.plot(
        x,
        map_model.predict(phi),
        color=next(colors),
        linestyle=next(line_styles),
        linewidth=2)

    if mle_model is not None:
        plt.plot(
            x,
            mle_model.predict(phi),
            color=next(colors),
            linestyle=next(line_styles),
            linewidth=2)
        legends.append('MLE')

    for lm in dist.sample(num_samples):
        plt.plot(
            x,
            lm.predict(phi),
            color=next(colors),
            linestyle=next(line_styles),
            linewidth=2)

    plt.legend(legends)
    return plt


def plot_policy(x, policy, colors):
    plt.stackplot(x, policy.T, colors=colors)
    plt.xlabel('feature')
    plt.ylabel('cumulative weight')
    plt.margins(0, 0)
    return plt


def plot_reward_across_features(x, methods):
    for name, method in methods.items():
        plt.plot(x, method['evs'], linewidth=2, label=name, **method['style'])
    plt.xlabel('feature')
    plt.ylabel('expected reward')
    if len(methods) > 1:
        plt.legend()
    return plt


class NamedStyles(object):
    def __init__(self):
        self.data = {}
        self._line_styles = line_style_table()
        self._colors = tableu20_color_table()

    def add(self, name):
        if name not in self.data:
            self.data[name] = {
                'linestyle': next(self._line_styles),
                'color': next(self._colors)
            }
        return self

    def __getitem__(self, name):
        if name not in self.data:
            self.add(name)
        return self.data[name]

#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Mar 28, 2018"
__version__ = "v3.1"

"""
Module contains class AwesomePlot.

Derivative of the matplotlib module. The aim is to create
visually attractive, unambigous and colour-blind-friendly
images, customised for PIK corporate design.

The class has the no instance variables.

Overriden inherited methods::

    (NoneType)  show            : future changes to pyplot.show()

"""

#TODO: distribution-adjusted colormap
#TODO: magic function to modify style of existing figures
#TODO: label on bar in histogram
#TODO: map plots

# Import NumPy for the array object and fast numerics.
import numpy as np

# import matplotlib and submodules
import matplotlib
from matplotlib import pyplot
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, axes_size, ImageGrid

from cycler import cycler
from itertools import cycle

# import warnings module to issue warnings on user input without interrupting the program
import warnings

# TODO: optional fig,ax objects as arguments: add_*(..., figax=None): fig, ax = figax


class AwesomeFigure(pyplot.Figure):

    def __init__(self, type="paper", **kwargs):

        if type == "paper":
            figsize = (6.75, 5)
        else:
            figsize = (1, 1)

        super(self.__class__, self).__init__(
                figsize=figsize,
                dpi=300,
                facecolor=None,
                edgecolor=None,
                linewidth=2,
                frameon=True,
                subplotpars=None,
                tight_layout=False,
                constrained_layout=True
        )

        self._set_style(type)

    def axes_constructor(self):
        pass


    def _set_style(self, type):

        default = {
            'axes.linewidth': .75,
            "axes.titlesize": 12,
            "grid.linewidth": 1,
            "lines.linewidth": 0.75,
            "patch.linewidth": .3,
            "lines.markersize": 5,
            "lines.markeredgewidth": 0,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.width": .5,
            "ytick.minor.width": .5,
            "xtick.major.pad": 7,
            "ytick.major.pad": 7
        }

        if type == "paper":
            pyplot.style.use("seaborn-paper")
            self._update_params({k: v * 2 for k, v in default.items()})
            self._update_params({"font.size": 14, "legend.fontsize": 14, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})

        self._update_params({'xtick.direction': 'in', 'ytick.direction': 'in', 'verbose.level': 'helpful'})

    def _update_params(self, dic):
        assert all([key in matplotlib.rcParams.keys() for key in dic.keys()])
        for key, val in dic.iteritems():
            matplotlib.rcParams[key] = val

    def show(self):
        pyplot.show()


class Plot(object):
    """
    AwesomePlot class.

    The class is consistently sets reasonable matplotlib rc parameters for three different use cases.
    It can be instantiated either to create images for

    a) publications (A4, pdf)
    b) talks (A5, png)
    c) icons (not implemented)

    Images are landscape per default.
    """

    def __init__(self, style=None, rc_spec={}, font_scale=2):
        """
            Initialise an instance of AwesomePlot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """

        if style is None:
            style = "seaborn-paper"

        pyplot.style.use(style)

        self.figure_format = "pdf"
        self.transparent = False





        if rc_spec:
            self._update_params(rc_spec)

        # predefine colour maps:
        self._custom_cmaps()
        self._set_default_colours('PIK')

        # linestyle sequence for multiplots
        self.linestyles = cycler("linestyle", ['-', '--', '-.', ':'])
        # marker style  sequence for multiplots
        self.markers = cycler("marker", ['o', 's', 'x', 'v', '+', '*'])

        self.figures = []

        # function aliases for backwards compatibility
        # self.add_contour = self.contour
        # self.add_distplot = self.distplot
        # self.add_hist = self.hist
        # self.add_lineplot = self.lineplot
        # self.add_network = self.network
        # self.add_scatterplot = self.scatterplot

    ###############################################################################
    # ##                       INTERNAL FUNCTIONS                              ## #
    ###############################################################################

    def _custom_cmaps(self):

        # generic discrete cmap (10 items)
        discrete_colours = ListedColormap(
                np.array(
                        ['#1f77b4', '#33a02c', '#ff7f00', '#6a3d9a', '#e31a1c', '#a6cee3', '#b2df8a', '#fb9a99',
                         '#fdbf6f', '#cab2d6']),
                'discrete'
        )
        register_cmap('discrete', cmap=discrete_colours)

        # PIK discrete cmap (4 items)
        pik_colours = ListedColormap(
                np.array(['#F25B28', '#009FDA', '#69923A', '#686C70']),
                'PIK'
        )
        register_cmap(pik_colours.name, cmap=pik_colours)

        # linear interpolated cmap (based on PIK colours)
        lin_colours = LinearSegmentedColormap.from_list(
                'linearPIK', [(0, 'w'), (1, hex2color('#e37222'))]
        )
        lin_colours.set_bad(hex2color('#8e908f'))
        register_cmap(lin_colours.name, cmap=lin_colours)

        # symmetric interpolated cmap (based on PIK colours)
        sym_colours = LinearSegmentedColormap.from_list(
                'divergingPIK', [(0, hex2color('#009fda')), (0.5, hex2color('#8e908f')), (1, hex2color('#e37222'))]
        )
        sym_colours.set_bad('k')
        register_cmap(sym_colours.name, cmap=sym_colours)

    def _set_default_colours(self, cmap_name):
        self.dfcmp = pyplot.get_cmap(cmap_name)
        self._update_params({'image.cmap': self.dfcmp.name})

    def _update_params(self, dic):
        assert all([key in matplotlib.rcParams.keys() for key in dic.keys()])
        for key, val in dic.iteritems():
            matplotlib.rcParams[key] = val

    def _pi_ax_yaxis(self, _ax):
        y_label = np.empty(np.size(_ax.get_yticks()), dtype='object')
        for i in range(np.size(_ax.get_yticks())):
            y_label[i] = str(round(_ax.get_yticks()[i] / np.pi, 1)) + "$\pi$"
        _ax.set_yticklabels(y_label)

    def _pi_ax_yaxis(self, _ax):
        x_label = np.empty(np.size(_ax.get_xticks()), dtype='object')
        for i in range(np.size(_ax.get_xticks())):
            x_label[i] = str(round(_ax.get_xticks()[i] / np.pi, 1)) + "$\pi$"
        _ax.set_xticklabels(x_label)


    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################


    def lineplot(self, x=None, lines={}, symmetric_error={}, labels=['x', 'y'], marker="o", sortfunc=None, grid=False,
                 legend=True, cmap=None, fig=None):
        """
        Plots (multiple) lines with optional shading.

        This function adds a matplotlib figure object to the figure collection of
        an AwesomePlot instance.

        Parameters
        ----------
        x: numpy ndarray (optional)
            array with x-values common to all lines
        lines:  dict
            dictionary of type {key: y} containing y-values,
            multiple lines need to be distinguished by unique keys
        symmetric_error: dict (optional)
            dictionary of type {key: [y - l, y +  u]} containing upper and lower
            intervals to indicate uncertainty, confidence intervals etc.
        labels: list [str]
            list containing  meaningful axis labels
        sortfunc: function or lambda expression
            optionally supply a function that is used to sort the line keys for plotting
            e.g. (a) sortfunc = float (b) sortfunc=f, where f = lambda x: float(x.split()[-2])
        grid: bool
            if true, background grid is drawn
        """

        assert len(labels) == 2

        if x is None:
            x = np.arange(len(lines[lines.keys()[0]]))

        if cmap is None:
            colour_cycler = cycler("color", self.dfcmp.colors)
        else:
            colour_cycler = cycler("color", pyplot.get_cmap(cmap).colors)
        style = self.linestyles * colour_cycler

        if fig is None:
            fig = pyplot.figure(constrained_layout=True)
            self.figures.append(fig)

        ax = fig.add_subplot(1, 1, 1)

        keys = sorted(lines.keys(), key=sortfunc, reverse=False)

        for i, sty in zip(keys, cycle(style)):
            # giving x as a dict, each line can have it's own x-axis
            _x = x[i] if isinstance(x, dict) else x
            if symmetric_error and i in symmetric_error.keys():
                shade = ax.fill_between(_x,
                                        lines[i] - symmetric_error[i],
                                        lines[i] + symmetric_error[i],
                                        alpha=0.3,
                                        edgecolor='none',
                                        facecolor=hex2color('#8E908F'),
                                        )
                ax.plot(_x, lines[i], marker=marker, mec=shade._facecolors[0], label=i, **sty)
            else:
                ax.plot(_x, lines[i], marker=marker, mec='w', label=i, **sty)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if grid:
            ax.set_gid()

        if legend:
            ax.legend(frameon=True)

        return fig, ax

    def scatterplot(self, x, y, labels=['x', 'y'], nbins=20, aspect="auto", normed=True, diag=True, legend=True, sortfunc=None, cmap=None):
        assert len(labels) == 2

        if isinstance(y, dict):
            if isinstance(x, dict):
                assert x.keys() == y.keys()
            else:
                x = {k: x for k in y.keys()}
        else:
            if isinstance(x, dict):
                warnings.warn("Please switch x and y!")
            else:
                x = {0: x}
                y = {0: y}
                legend = False

        types = len(y.keys())

        if cmap is None:
            colour_cycler = cycler("color", self.dfcmp.colors)
        else:
            colour_cycler = cycler("color", pyplot.get_cmap(cmap).colors)
        style = self.markers * colour_cycler

        fig, ax = pyplot.subplots(2, 2, figsize=(6, 6), gridspec_kw={'width_ratios':[3, 1], 'height_ratios':[1,3]}, constrained_layout=True)
        self.figures.append(fig)
        fig.subplots_adjust(left=0.2, right=0.9, top=.99, bottom=0.15, hspace=0., wspace=0.)

        ax[1, 0].set_aspect(aspect)

        ax[0, 0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[0, 1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1, 1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        #TODO: ticks in/out

        ax[1, 0].set_xlabel(labels[0])
        ax[1, 0].set_ylabel(labels[1])

        keys = sorted(y.keys(), key=sortfunc, reverse=False)

        if normed:
            for k in keys:
                x[k] = (x[k] - x[k].mean(axis=0)) / x[k].std(axis=0)
                y[k] = (y[k] - y[k].mean(axis=0)) / y[k].std(axis=0)

        colours = []
        for i, sty in zip(keys, cycle(style)):
            ax[1, 0].scatter(x[i], y[i], label=i, **sty)
            colours.append(sty['color'])

        bins = np.linspace(np.min(x.values()), np.max(x.values()), nbins)
        ax[0, 0].hist(x.values(), bins=bins, color=colours, density=1, histtype='bar', stacked=True)

        bins = np.linspace(np.min(y.values()), np.max(y.values()), nbins)
        ax[1, 1].hist(y.values(), bins=bins, color=colours, orientation='horizontal', density=1, histtype='bar', stacked=True)

        ax[0, 1].axis("off")
        if legend:
            h, l = ax[1, 0].get_legend_handles_labels()
            ncol = 2 if types > 6 else 1
            ax[0, 1].legend(h, l, frameon=True, loc=3, ncol=ncol, fontsize=9)

        if diag:
            ax[1, 0].plot([np.min(x.values()), np.max(x.values())], [np.min(x.values()), np.max(x.values())], "k:", zorder=-1)

        return fig, ax

    def snapshot(self, y1, labels=['x', 'y1'], y1_base=None, pi=False, cmap=None, fig=None):
        assert len(labels) == 2

        n = len(y1)

        x = range(n)

        if fig is None:
            fig = pyplot.figure(constrained_layout=True)
            self.figures.append(fig)

        ax = fig.add_subplot(1, 1, 1)

        if cmap is None:
            cc = self.dfcmp.colors
        else:
            cc = pyplot.get_cmap(cmap).colors

        ax.plot(x, y1, linewidth=0, marker=".", label=labels[1], mfc=cc[0])
        if y1_base is not None:
            ax.hlines(y=y1_base, xmin=0, xmax=n - 1, zorder=-1, linestyle=':')

        if pi == "y1":
            self._pi_ax_yaxis(ax)


        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        return fig, ax

    def double_snapshot(self, y1, y2, labels=['x', 'y1', 'y2'], y1_base=None, y2_base=None, pi="y1", cmap=None):
        assert len(labels) == 3

        n = max(len(y1), len(y2))

        x = range(n)

        fig, ax = pyplot.subplots(nrows=2, ncols=1, sharex=True)
        self.figures.append(fig)
        fig.subplots_adjust(left=0.2, right=0.99, top=.99, bottom=0.2, hspace=0.1, wspace=0.)

        if cmap is None:
            cc = self.dfcmp.colors
        else:
            cc = pyplot.get_cmap(cmap).colors

        ax[0].plot(x, y1, linewidth=0, marker=".", label=labels[1], mfc=cc[0])
        if y1_base is not None:
            ax[0].hlines(y=y1_base, xmin=0, xmax=n - 1, zorder=-1, linestyle=':')
        ax[1].plot(x, y2, linewidth=0, marker=".", label=labels[2], mfc=cc[1])
        if y2_base is not None:
            ax[1].hlines(y=y2_base, xmin=0, xmax=n - 1, zorder=-1, linestyle=':')

        if pi == "y1":
            self._pi_ax_yaxis(ax[0])
        if pi == "y2":
            self._pi_ax_yaxis(ax[1])
        else:
            pass

        ax[1].set_xlabel(labels[0])
        ax[0].set_ylabel(labels[1])
        ax[1].set_ylabel(labels[2])

        return fig, ax

    def show(self):
        if len(self.figures) > 0:
            pyplot.show()
        else:
            pass
            #raise ValueError("No open figures to plot.")

if __name__ == "__main__":
    import numpy as np
    p = pyplot.figure(FigureClass=AwesomeFigure, type="paper")
    x = np.arange(10)
    ax = p.add_subplot(111, aspect="auto")
    ax.scatter(1, 2, label="test")
    ax.plot(x, x**2, label="quad")
    ax.legend()
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    p.show()
    p.savefig("test")

#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "May 23, 2016"
__version__ = "v0.3"

"""
Module contains class Plot.

Derivative of the matplotlib module. The aim is to create
visually attractive, unambigous and colour-blind-friendly
images, customised for PIK corporate design.

The class has the no instance variables.

Overriden inherited methods::

    (NoneType)  show            : future changes to pyplot.show()

"""

# Import NumPy for the array object and fast numerics.
import numpy as np

# import matplotlib and submodules
import matplotlib
from matplotlib import pyplot
from matplotlib import cycler
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color

# import seaborn for fancy style templates
import seaborn

# import warnings module to issue warning on user input without interrupting the program
import warnings

class Plot(object):
    """
    Plot class.

    The class is consistently sets reasonable matplotlib rc parameters for three different use cases.
    It can be instantiated either to create images for

    a) publications (A4, pdf)
    b) talks (A5, png)
    c) icons (not implemented)

    Images are landscape per default.
    """

    # predefine colour maps:

    # generic discrete cmap (10 items)
    discrete_colours = ListedColormap(
        np.array(
            ['#1f77b4', '#33a02c', '#ff7f00', '#6a3d9a', '#e31a1c', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6']),
        'discrete'
    )
    register_cmap('discrete', cmap=discrete_colours)

    # PIK discrete cmap (4 items)
    pik_colours = ListedColormap(
        np.array(['#F25B28', '#009FDA', '#69923A', '#686C70']),
        'pik'
    )
    register_cmap(pik_colours.name, cmap=pik_colours)

    # linear interpolated cmap (based on PIK colours)
    lin_colours = LinearSegmentedColormap.from_list(
        'linear', [(0, 'w'), (1, hex2color('#e37222'))]
    )
    lin_colours.set_bad(hex2color('#8e908f'))
    register_cmap(lin_colours.name, cmap=lin_colours)

    # symmetric interpolated cmap (based on PIK colours)
    sym_colours = LinearSegmentedColormap.from_list(
        'sym', [(0, hex2color('#009fda')), (0.5, hex2color('#8e908f')), (1, hex2color('#e37222'))]
    )
    sym_colours.set_bad('k')
    register_cmap(sym_colours.name, cmap=sym_colours)

    # linestyle sequence for multiplots
    linestyles = np.tile(['-', '--', '-.', ':'], 1 + discrete_colours.N // 4)[:discrete_colours.N]

    def __init__(self, output='paper', rc_spec={}, font_scale=1.1):
        """
            Initialise an instance of Plot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """

        assert output in ["paper", "talk", "poster", "notebook"]

        # workaround for KeyError in self.rc savefig.format
        if output in ["talk", "poster"]:
            self.figure_format = "png"
            self.transparent = True
        else:
            self.figure_format = "pdf"
            self.transparent = False

        self.rc = {'xtick.direction': 'in',
                   'ytick.direction': 'in',
                   'verbose.level': 'helpful',
                   'lines.linewidth': 3,
                   'axes.linewidth': 3
                   }

        if rc_spec:
            self.rc.update(rc_spec)

        seaborn.set_style(style="white", rc=self.rc)
        seaborn.set_context(output, font_scale=font_scale, rc=self.rc)

        self.set_default_colours('pik')

        self.figures = []

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################


    def add_lineplot(self, x=None, lines={}, shades={}, labels=['x', 'y'], sortfunc=None, grid=False, layout=True):
        """
        Plots (multiple) lines with optional shading.

        This function adds a matplotlib figure object to the figure collection of
        an Plot instance.

        Parameters
        ----------
        x: numpy ndarray (optional)
            array with x-values common to all lines
        lines:  dict
            dictionary of type {key: y} containing y-values,
            multiple lines need to be distinguished by unique keys
        shades: dict (optional)
            dictionary of type {key: [y - l, y +  u]} containing upper and lower
            intervals to indicate uncertainty, confidence intervals etc.
        labels: list [str]
            list containing  meaningful axis labels
        sortfunc: function or lambda expression
            optionally supply a function that is used to sort the line keys for plotting
            e.g. (a) sortfunc = float (b) sortfunc=f, where f = lambda x: float(x.split()[-2])
        grid: bool
            if true, background grid is drawn
        layout: bool
            if false min and max will not be set , important for plots with NANs and Infs
        """

        assert len(labels) == 2
        # the next line leads to an error if there are more lines to be plotted
        # than colours available, although there are alsy different linestyles possible
        # TODO write new assert with message that you can use other colourmaps by p.set_default_colours(x),
        # with x = 'discrete'(10 colours), 'pik' (4 colours, default), 'linear', 'sym'
        #assert len(lines.keys()) <= self.dfcmp.N

        if x is None:
            x = np.arange(len(lines[0]))

        if shades:
            assert sorted(shades.keys()) == sorted(lines.keys())

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        # determine boundaries
        if layout:
            xmin = np.min(x)
            xmax = np.max(x)
            if not shades:
                ymin = np.min([np.min(l) for l in lines.itervalues()])
                ymax = np.max([np.max(l) for l in lines.itervalues()])
            else:
                ymin = np.min([np.min(l) for l in shades.itervalues()])
                ymax = np.max([np.max(l) for l in shades.itervalues()])

            xmargin = (xmax - xmin) / 200.
            ymargin = (ymax - ymin) / 200.

            scale = np.log(1 + 1. * (xmax - xmin) / len(x))

            ax.axis([xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin])

        if grid:
            ax.grid()

        for i in sorted(lines.keys(), key=sortfunc, reverse=True):
            if shades:
                shade = ax.fill_between(x, shades[i][0], shades[i][1], alpha=0.3, edgecolor='none',
                                        facecolor=hex2color('#8E908F'))
                ax.plot(x, lines[i], marker='o', mew=3.*scale, mec=shade._facecolors[0], ms=10.*scale, label=i)
            else:
                if layout:
                    ax.plot(x, lines[i], marker='o', mec='w', mew=3*scale, ms=10.*scale, label=i)
                else:
                    ax.plot(x, lines[i], marker='o', mec='w', label=i)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        pyplot.legend(frameon=True)
        fig.tight_layout()

        self.figures.append(fig)

        return fig

    def add_distplot(self, x, y, labels=['x', 'y'], linestyle='-', filled=True, text=True):

        assert len(labels) == 2

        from scipy.stats.mstats import mquantiles

        m = np.mean(y, axis=1)
        q = mquantiles(y, axis=1, prob=list([.05, .25, .75, .95]))

        # determine boundaries
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = y.min()
        ymax = y.max()

        xmargin = (xmax - xmin) / 200.
        ymargin = (ymax - ymin) / 200.

        scale = np.log(1 + .5 * (xmax - xmin) / len(x))

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        ax.axis([xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin])

        if filled:
            ax.fill_between(x, q[:, 0],
                            q[:, 3], facecolor=hex2color('#8E908F'), edgecolor='none', alpha=0.3)
            ax.fill_between(x, q[:, 1],
                            q[:, 2], facecolor=hex2color('#8E908F'), edgecolor='none', alpha=0.3)
        else:
            ax.plot(x, q[:, 1], color='r', linestyle='--', label='25\%')
            ax.plot(x, q[:, 2], color='r', linestyle='--', label='75\%')
            ax.plot(x, q[:, 0], color='r', linestyle=':', label='5\%')
            ax.plot(x, q[:, 3], color='r', linestyle=':', label='95\%')

        if text:
            ax.text(x[1], q[1, 0] * .95, r'5%') #, fontsize=.6 * self.textsize)
            ax.text(x[2], q[2, 1] * .95, r'25%') #, fontsize=.6 * self.textsize)
            ax.text(x[3], q[3, 2] * .95, r'75%') #, fontsize=.6 * self.textsize)
            ax.text(x[4], q[4, 3] * .95, r'95%') #, fontsize=.6 * self.textsize)

        ax.plot(x, m, marker='o', mec='w', mew=3*scale, ms=10*scale)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        fig.tight_layout()

        self.figures.append(fig)

        return fig

    def add_contour(self, x, y, z, labels=['x', 'y', 'z'], nlevel=10, sym=False, text=False, horizontal=False, pi=None, layout=True, fixed_scale=None):
        """
            Plots Contourplots

            Parameters
            ----------
            x: array
                x-values associated to the entries in z
            y: array
                y-values associated to the entries in z
            z: matrix
                data of shape [len(x), len(y)] containing values for all (x, y) pairs
            nlevel: int
                number of levels of the contourplot
            sym: bool
                False: using linear colour scale, else diverging
            text: bool
                True: labels on every level line
            horizontal: bool
                True: horizontal colour bar
            pi: "xaxis" or "yaxis"
                if one of the axis is given in multiples of pi
            layout: bool
                False means, the contourf/contour function dont get a
                number of levels or a zmin and zmax. This is necessary
                for a matrix z with NANs or Infs in it, since then
                zmin and zmax become NAN/Inf.
            fixed_scale: tuple
                min/max values to apply a fixed colour scale to z-values

        """
        assert len(labels) == 3

        # Issue warning if z contains NaN or Inf
        if not np.isfinite(z).all():
            warnings.warn("Since z is not finite, it would be better to use layout=False.")

        if sym:
            cmap = pyplot.get_cmap('sym')
        else:
            cmap = pyplot.get_cmap('linear')

        backup = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 1

        # determine boundaries

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()

        if not fixed_scale is None:
            zmin, zmax = fixed_scale
        else:
            zmin = np.floor(z[np.isfinite(z)].min())
            zmax = z[np.isfinite(z)].max()

            if zmax > 0.5:
                zmax = np.ceil(z[np.isfinite(z)].max())

            if zmin == zmax:
                zmax += 0.5

        pyplot.gca().patch.set_color('#8e908f')  # print the Nan/inf Values in black)

        if layout:
            levels = np.linspace(zmin, zmax, nlevel + 1, endpoint=True)
            c = ax.contourf(x, y, z, levels=levels, cmap=cmap, origin='lower', antialiased=True, vmin=zmin, vmax=zmax)
            cl = ax.contour(x, y, z, colors='k', levels=levels)
        else:
            c = ax.contourf(x, y, z, cmap=cmap, origin='lower', antialiased=True, vmin=zmin, vmax=zmax)
            cl = ax.contour(x, y, z, colors='k')

        if text:
            ax.clabel(cl, fontsize=.25 * self.textsize, inline=1)

        ax.axis([xmin, xmax, ymin, ymax])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if pi == "xaxis":
            x_label = np.empty(np.size(ax.get_xticks()), dtype='object')
            for i in range(np.size(ax.get_xticks())):
                x_label[i] = str(ax.get_xticks()[i]) + "$\pi$"
            ax.set_xticklabels(x_label)

        if pi == "yaxis":
            y_label = np.empty(np.size(ax.get_yticks()), dtype='object')
            for i in range(np.size(ax.get_yticks())):
                y_label[i] = str(ax.get_yticks()[i]) + "$\pi$"
            ax.set_yticklabels(y_label)

        if horizontal:
            fig.colorbar(c, label=labels[2], orientation='horizontal', pad=0.2)
        else:
            fig.colorbar(c, label=labels[2])  # not so cool for smalll numbers format=r"%.1f"

        self.figures.append(fig)

        matplotlib.rcParams['lines.linewidth'] = backup

        return fig


    def add_scatterplot(self, x, y, labels=['x', 'y'], factor=None, show_annot=None, bins=20, kind="scatter", kdeplot=False, c_map="linear"):
        assert len(labels) == 2

        if isinstance(x, dict):
            assert sorted(x.keys()) == sorted(y.keys())
            assert len(x.keys()) <= self.dfcmp.N
            # determine boundaries
            xmin = np.min([np.min(l) for l in x.itervalues()])
            xmax = np.max([np.max(l) for l in x.itervalues()])
            ymin = np.min([np.min(l) for l in y.itervalues()])
            ymax = np.max([np.max(l) for l in y.itervalues()])
        else:
            # determine boundaries
            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()

        # adjust colors
        if kdeplot:
            c = "k"
        else:
            if factor is None:
                c = self.pik_colours.colors[0]
            else:
                c = factor

        settings = {
            "joint_kws": dict(alpha=1, c=c, cmap=pyplot.get_cmap(c_map)),
            "marginal_kws": dict(bins=bins, rug=False),
            "annot_kws": dict(stat=None, frameon=True, loc="best", handlelength=0),
            "space": 0.1,
            "kind": kind,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax)
        }

        scatter = seaborn.jointplot(x, y, stat_func=show_annot, **settings)

        if kdeplot:
            scatter.plot_joint(seaborn.kdeplot, shade=True, cut=5, zorder=0, n_levels=6, cmap=pyplot.get_cmap(c_map))

        scatter.set_axis_labels(*labels)

        self.figures.append(scatter.fig)

        return scatter.fig

    def add_hist(self, data, label='x', nbins=20):

        # ensure data is nested list
        if isinstance(data[0], (int, float)):
            data = list([data, ])

        xmin = np.min([np.min(l) for l in data])
        xmax = np.max([np.max(l) for l in data])

        xmargin = (xmax - xmin) / 100.

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()

        bottom = np.zeros(nbins)
        ymax = 0.
        counter = 0
        _, b = np.histogram(data[0], bins=nbins, density=True)
        for d in data:
            c = list(matplotlib.rcParams['axes.prop_cycle'])[counter]['color']
            h, _ = np.histogram(d, bins=nbins, density=True)
            ax.bar(.5 * (b[1:] + b[:-1]), h, bottom=bottom, color=c, edgecolor="none", align='center', zorder=1,
                   width=(xmax - xmin) / (nbins * 1.5))
            bottom += h
            counter += 1
            ymax += h.max()

        ax.set_xlim([xmin - xmargin, xmax + xmargin])
        ax.set_ylim([0., ymax * 1.1])
        ax.set_xlabel(label)
        ax.set_ylabel(r'density')

        ax.yaxis.grid(color='w', linestyle='-', zorder=2)

        self.figures.append(fig)

        return fig

    def add_network(self, adjacency, styles={}, sym=True, axis_labels=None, vertex_labels=None, labels=False, height=False):
        """
            Plots network, submit eg vertex color values via styles={"vertex_color":values}

            Parameters
            ----------
            adjacency:
            styles: dict

            sym: bool
                if true:
            axis_labels:

            vertex_labels: array
                e.g. pars["input_power"]
            labels: bool

            height: bool


        """
        if height:
            from mpl_toolkits.mplot3d import Axes3D
        from scipy.sparse import issparse, isspmatrix_dok

        if issparse(adjacency):
            assert isspmatrix_dok(adjacency)
            # print "Build network from sparse dok matrix."
            N = adjacency.shape[0]
            edgelist = sorted(set([tuple(np.sort(key)) for key in adjacency.iterkeys()]))
        else:
            N = len(adjacency)
            edgelist = np.vstack(np.where(adjacency > 0)).transpose()
            edgelist = sorted(set([tuple(np.sort(edgelist[i])) for i in range(len(edgelist))]))

        if sym:
            cmap = pyplot.get_cmap("sym")
        else:
            cmap = pyplot.get_cmap("linear")

        visual_style = dict(
            edge_color=np.repeat('#8e908f', len(edgelist)),
            edge_width=seaborn.axes_style()["axes.linewidth"],
            vertex_size=100,
            vertex_label=range(N)
        )

        if styles:
            visual_style.update(styles)

        if not visual_style.has_key("layout"):
            if height:
                visual_style["layout"] = np.random.random([N, 3])
            else:
                visual_style["layout"] = np.random.random([N, 2])
            print "Assign random layout for plotting."

        if visual_style.has_key("edge_color_dict"):
            f = lambda x: visual_style["edge_color_dict"][x]
            for i, e in enumerate(edgelist):
                visual_style["edge_color"][i] = f(e)

        if height:
            fig = pyplot.figure()
            ax = fig.gca(projection='3d')
            x, y, z = zip(*visual_style["layout"])
            args = (x, y, z)
        else:
            fig, ax = pyplot.subplots(nrows=1, ncols=1)
            fig.tight_layout()
            x, y = zip(*visual_style["layout"])
            args = (x, y)

        # ax.axis("off")

        for i, e in enumerate(edgelist):
            edge = np.vstack((visual_style["layout"][e[0]], visual_style["layout"][e[1]]))
            if height:
                xyz = edge[:, 0], edge[:, 1], edge[:, 2]
            else:
                xyz = edge[:, 0], edge[:, 1]
            ax.plot(*xyz,
                    color=visual_style["edge_color"][i],
                    linestyle='-',
                    lw=visual_style["edge_width"],
                    alpha=0.5,
                    zorder=1)

        margin = max(0.05 * (np.max(x) - np.min(x)), 0.05 * (np.max(y) - np.min(y)))
        ax.set_xlim([np.min(x) - margin, np.max(x) + margin])
        ax.set_ylim([np.min(y) - margin, np.max(y) + margin])

        if not visual_style.has_key("vertex_color"):
            nodes = ax.scatter(*args,
                               c='#8e908f',
                               s=visual_style["vertex_size"],
                               cmap=cmap,
                               edgecolor='w',
                               zorder=2)
        else:
            nodes = ax.scatter(*args,
                               c=visual_style["vertex_color"],
                               s=visual_style["vertex_size"],
                               cmap=cmap,
                               vmin=np.floor(np.min(visual_style["vertex_color"])),
                               vmax=np.ceil(np.max(visual_style["vertex_color"])),
                               edgecolor='w',
                               zorder=2)
            cb = fig.colorbar(nodes, orientation='horizontal', shrink=0.66, format=r"%.1f")
            # deprecated
            # [t.set_fontsize(seaborn.axes_style()["legend.fontsize"]) for t in cb.ax.get_xticklabels()]

        if labels:
            for i in xrange(N):
                pyplot.annotate(str(i), xy=(x[i], y[i]), xytext=(3, 3), textcoords='offset points',
                                # size=0.5 * self.rc["font.size"],
                                horizontalalignment='left', verticalalignment='bottom')

        if axis_labels:
            ax.set_xlabel(axis_labels[0], labelpad=30)
            ax.set_ylabel(axis_labels[1], labelpad=30)
            if height:
                ax.set_zlabel(axis_labels[2], labelpad=30)

        if vertex_labels is not None:
            for i in xrange(N):
                pyplot.annotate(np.around(vertex_labels[i], decimals=3), xy=(x[i], y[i]), xytext=(3, -25),
                                textcoords='offset points',
                                # size=0.5 * self.params["font.size"],
                                horizontalalignment='left', verticalalignment='bottom')

        # we may adjust the background colour to make light nodes more visible
        #ax.set_axis_bgcolor((.9, .9, .9))

        self.figures.append(fig)

        return fig

    def save(self, fnames, fig = None):
        if fig:
            fig.savefig(filename=fnames + '.' + self.figure_format, bbox_inches='tight', transparent=self.transparent)
            self.clear(fig)
        else:
            assert len(fnames) == len(self.figures)
            for i, fig in enumerate(self.figures):
                print "save:", fnames[i] + '.' + self.figure_format
                fig.savefig(filename=fnames[i] + '.' + self.figure_format, bbox_inches='tight', transparent=self.transparent)
                pyplot.close(fig)
            for i, fig in enumerate(self.figures):
                self.figures.remove(fig)


    def clear(self, fig):
        assert isinstance(fig, pyplot.Figure)
        pyplot.close(fig)
        self.figures.remove(fig)


    def show(self):
        if len(self.figures) > 0:
            pyplot.show()
        else:
            pass
            #raise ValueError("No open figures to plot.")


    def show_cmap(self, cmap_name):
        if isinstance(cmap_name, str):
            cmap = pyplot.get_cmap(cmap_name)
        else:
            cmap = cmap_name
        a=np.outer(np.arange(0,1,0.01), np.ones(10))
        pyplot.figure(figsize=(2,10))
        pyplot.axis("off")
        pyplot.imshow(a,aspect='auto',interpolation='nearest', cmap=cmap)
        pyplot.title(cmap_name,fontsize=15)
        pyplot.show()

    def update_params(self, dic):
        assert all([key in matplotlib.rcParams.keys() for key in dic.keys()])
        self.rc.update(dic)
        # do not use seaborn.set(), this overrides all rc params ...
        for key, val in dic.iteritems():
            matplotlib.rcParams[key] = val


    def portrait(self):
        canvas = self.rc['figure.figsize']
        if canvas[1] > canvas[0]:
            warnings.warn("Figure is already in portrait orientation.")
        else:
            self.rc['figure.figsize'] = canvas[::-1]

    def set_default_colours(self, cmap_name):
        self.dfcmp = pyplot.get_cmap(cmap_name)
        self.update_params(
            {
                'axes.prop_cycle': cycler('color', list(self.dfcmp.colors)) + \
                    cycler('linestyle', self.linestyles[:self.dfcmp.N]),
                'image.cmap': self.dfcmp.name
            }
        )
        #TODO color cycler with selected colours

    def set_log(self, fig, log="y"):
        assert fig in self.figures
        if log == "y":
            fig.axes[0].set_yscale('symlog')
        elif log == "x":
            fig.axes[0].set_xscale('symlog')
        elif log == "xy":
            fig.axes[0].set_xscale('symlog')
            fig.axes[0].set_yscale('symlog')
        else:
            raise ValueError("Invalid input. Must be x, y, or xy.")

if __name__ == "__main__":
    p = Plot(output="paper")

    #p.show_cmap(p.dfcmp.name)

    labels = [r'$\phi$']

    x = np.arange(10)
    z = 1 - 2. * np.random.random([10, 10])
    #p.add_contour(x, x, z, sym=True)

    import networkx as nx
    A = nx.to_scipy_sparse_matrix(nx.erdos_renyi_graph(10, 0.01), format="dok")


    f = p.add_network(A, styles={"vertex_color": np.random.random(10)}, height=False)

    p.save("test/test", f)

    p.show()


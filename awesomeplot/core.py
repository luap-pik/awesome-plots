#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "May 23, 2016"
__version__ = "v0.2"

"""
Module contains class AwesomePlot.

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

class AwesomePlot(object):
    """
    AwesomePlot class.

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
        np.array(['#e37222', '#009fda', '#69923a', '#8e908f']),
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

    def __init__(self, output='paper'):
        """
            Initialise an instance of AwesomePlot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """

        self.set_default_colours('pik')

        # base parameters
        self.params = {}

        if output == 'paper':
            self.textsize = 40
            self.params['figure.figsize'] = (11.69, 8.268) # A4
            self.params['savefig.format'] = 'pdf'
            self.params['pdf.compression'] = 6  # 0 to 9
            self.params['pdf.fonttype'] = 42
            self.params['savefig.dpi'] = 300
        elif output == 'talk':
            self.textsize = 20
            self.params['figure.figsize'] = (8.268, 5.872) # A5
            self.params['savefig.format'] = 'png'
            self.params['savefig.dpi'] = 300
        elif output == 'icon':
            raise NotImplementedError('Simplified plots as icons for talks not implemented yet.')
        else:
            raise ValueError('Invalid image format. Either paper or talk!')

        common_params = {'xtick.labelsize': .9 * self.textsize,
                         'ytick.labelsize': .9 * self.textsize,
                         'xtick.major.size': 5,  # major tick size in points
                         'xtick.minor.size': 2,  # minor tick size in points
                         'ytick.major.size': 5,  # major tick size in points
                         'ytick.minor.size': 2,  # minor tick size in points
                         'xtick.major.width': 2,  # major tick size in points
                         'xtick.minor.width': .5,  # minor tick size in points
                         'ytick.major.width': 2,  # major tick size in points
                         'ytick.minor.width': .5,  # minor tick size in points
                         'xtick.major.pad': 8,  # distance to major tick label in points
                         'xtick.minor.pad': 8,
                         'ytick.major.pad': 4,  # distance to major tick label in points
                         'ytick.minor.pad': 4,
                         'xtick.direction': 'in',
                         'ytick.direction': 'in',
                         'axes.labelsize': self.textsize,
                         'axes.linewidth': 3,
                         'axes.unicode_minus': True,
                         'axes.formatter.use_mathtext': True,
                         'axes.prop_cycle': cycler('color', list(self.dfcmp.colors)) +
                                            cycler('linestyle', self.linestyles[:self.dfcmp.N]),
                         'axes.xmargin': 0.05,
                         'axes.ymargin': 0.05,
                         'axes.labelweight': 'bold',
                         'contour.negative_linestyle': 'dashed',
                         'lines.markersize': 10,  # size in points
                         'legend.fontsize': .6 * self.textsize,
                         'legend.numpoints': 2,
                         'legend.handlelength': 1.,
                         'legend.fancybox': True,
                         'lines.linewidth': 3,
                         'grid.linewidth': 1,
                         'image.cmap': self.dfcmp.name,
                         # 'figure.autolayout': True,
                         'font.family': 'serif',
                         'font.serif': 'cm10',
                         'font.size': self.textsize,
                         'font.weight': 'bold',
                         'text.usetex': True,
                         'text.latex.preamble': r'\boldmath',
                         # 'savefig.transparent': True, # problems with transparency, e.g. in Inkscape?
                         'verbose.level': 'helpful'
                         }

        self.params.update(common_params)
        matplotlib.rcParams.update(self.params)

        self.figures = []

    @classmethod
    def paper(cls):
        """
        Class method yielding an AwesomePlot instance of type "paper"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """
        return cls(output='paper')

    @classmethod
    def talk(cls):
        """
        Class method yielding an AwesomePlot instance of type "talk"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """
        return cls(output='talk')

    @classmethod
    def icon(cls):
        """
        Class method yielding an AwesomePlot instance of type "icon"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """
        return cls(output='icon')

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    def add_lineplot(self, x=None, lines={}, shades={}, labels=['x', 'y'], grid=False):
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
        shades: dict (optional)
            dictionary of type {key: [y - l, y +  u]} containing upper and lower
            intervals to indicate uncertainty, confidence intervals etc.
        labels: list [str]
            list containing  meaningful axis labels
        grid: bool
            if true, background grid is drawn
        """

        assert len(labels) == 2
        assert len(lines.keys()) <= self.dfcmp.N

        if x is None:
            x = np.arange(len(lines[0]))

        if shades:
            assert sorted(shades.keys()) == sorted(lines.keys())

        # determine boundaries
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

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        ax.axis([xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin])
        if grid:
            ax.grid()
        for i in lines.keys():
            if shades:
                shade = ax.fill_between(x, shades[i][0], shades[i][1], alpha=0.3, edgecolor='none',
                                        facecolor=hex2color('#8E908F'))
                ax.plot(x, lines[i], marker='o', mew='3', mec=shade._facecolors[0])#, ms=12)
            else:
                ax.plot(x, lines[i], marker='o', mec='w', mew='3')#, ms=10)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.tight_layout()

        self.figures.append(fig)

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
            ax.text(x[1], q[1, 0], '5\%', fontsize=.6 * self.textsize)
            ax.text(x[2], q[2, 1], '25\%', fontsize=.6 * self.textsize)
            ax.text(x[3], q[3, 2], '75\%', fontsize=.6 * self.textsize)
            ax.text(x[4], q[4, 3], '95\%', fontsize=.6 * self.textsize)

        ax.plot(x, m, marker='o', mec='w', mew='3', ms=10)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        fig.tight_layout()

        self.figures.append(fig)

    def add_contour(self, x, y, z, labels=['x', 'y', 'z'], nlevel=10, sym=False, text=False):

        assert len(labels) == 3

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

        # r = int(np.log(max(abs(z.min()), abs(z.max()))))
        zmin = np.floor(z.min())
        zmax = np.ceil(z.max())
        levels = np.linspace(zmin, zmax, nlevel + 1, endpoint=True)

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()

        c = ax.contourf(x, y, z, levels=levels, cmap=cmap, origin='lower', antialiased=True, vmin=zmin, vmax=zmax)
        cl = ax.contour(x, y, z, colors='k', levels=levels)
        if text:
            ax.clabel(cl, fontsize=.25 * self.textsize, inline=1)

        ax.axis([xmin, xmax, ymin, ymax])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.colorbar(c, label=labels[2], format=r"%.1f")

        self.figures.append(fig)

        matplotlib.rcParams['lines.linewidth'] = backup

    def add_scatter(self, x={}, y={}, labels=['x', 'y'], bins=20):
        assert len(labels) == 2

        backup = matplotlib.rcParams['figure.figsize']
        matplotlib.rcParams['figure.figsize'] = (11.69, 11.69)

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

        xmargin = binwidthx = (xmax - xmin) / bins
        ymargin = binwidthy = (ymax - ymin) / bins

        fig = pyplot.figure()

        pyplot.ylabel(labels[0])
        pyplot.xlabel(labels[1])

        gs = matplotlib.gridspec.GridSpec(5, 5,
                                        wspace=0.0,
                                        hspace=0.0
                                        )

        axScatter = pyplot.subplot(gs[1:, :4])
        axHistx = pyplot.subplot(gs[0, :4], sharex=axScatter, frameon=False)
        axHisty = pyplot.subplot(gs[1:, 4], sharey=axScatter, frameon=False)

        # the scatter plot:
        axScatter.set_xlim((xmin - xmargin, xmax + xmargin))
        axScatter.set_ylim((ymin - ymargin, ymax + ymargin))
        axScatter.plot(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000), ':k')
        if isinstance(x, dict):
            axScatter.set_prop_cycle(matplotlib.cycler('marker', ['o', '<', 'd', 's', 'v', 'p', '>', '8', '*', '^']) +
                                     matplotlib.cycler('mfc', list(self.discrete_colours.colors)))
            for k in x.keys():
                axScatter.plot(x[k], y[k], lw=0., alpha=.25)
        else:
            axScatter.plot(x, y, 'o', lw=0., alpha=.25)
        axScatter.set_xlabel(labels[0])
        axScatter.set_ylabel(labels[1])

        # histograms
        if isinstance(x, dict):
            X = [item for sublist in x.values() for item in sublist]
            Y = [item for sublist in y.values() for item in sublist]
        else:
            X = x
            Y = y
        binsx = np.arange(xmin - binwidthx, xmax + binwidthx, binwidthx)
        binsy = np.arange(ymin - binwidthy, ymax + binwidthy, binwidthy)
        wx = np.ones_like(X) / float(len(X))
        wy = np.ones_like(Y) / float(len(Y))
        vx, _, _ = axHistx.hist(X, bins=binsx, weights=wx,
                                facecolor=self.dfcmp.colors[0], alpha=0.75)
        Xmax = np.ceil(vx.max() * 10.) / 10.
        axHistx.axis([xmin - xmargin, xmax + xmargin, 0., Xmax])
        vy, _, _ = axHisty.hist(Y, bins=binsy, weights=wy, orientation='horizontal',
                                facecolor=self.dfcmp.colors[0], alpha=0.75)
        Ymax = np.ceil(vy.max() * 10.) / 10.
        axHisty.axis([0., Ymax, ymin - ymargin, ymax + ymargin])

        # axes
        a, b = axHistx.get_xaxis().get_view_interval()
        c, d = axHistx.get_yaxis().get_view_interval()
        axHistx.add_artist(pyplot.Line2D((b, b), (c, d), color='k', linewidth=2*self.params['axes.linewidth']))
        a, b = axHisty.get_xaxis().get_view_interval()
        c, d = axHisty.get_yaxis().get_view_interval()
        axHisty.add_artist(pyplot.Line2D((a, b), (d, d), color='k', linewidth=2*self.params['axes.linewidth']))

        # ticks
        axScatter.get_xaxis().tick_bottom()
        axScatter.get_yaxis().tick_left()
        axHistx.get_xaxis().tick_bottom()
        axHistx.get_yaxis().tick_right()
        axHisty.get_xaxis().tick_top()
        axHisty.get_yaxis().tick_left()
        axHistx.set_yticks(np.linspace(0, Xmax, 3, endpoint=True)[1:])
        axHisty.set_xticks(np.linspace(0, Ymax, 3, endpoint=True)[1:])
        axHistx.tick_params(labelsize=.6*self.textsize)
        axHisty.tick_params(labelsize=.6*self.textsize)
        for tl in axHistx.get_xticklabels() + axHisty.get_yticklabels():
            tl.set_visible(False)

        self.figures.append(fig)

        matplotlib.rcParams['figure.figsize'] = backup

    def add_hist(self, data, label='x', nbins=20):

        # ensure data is nested list
        if isinstance(data[0], (int, float)):
            data = list([data,])

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
                   width=(xmax - xmin) / (nbins + 1))
            bottom += h
            counter += 1
            ymax += h.max()

        ax.set_xlim([xmin - xmargin, xmax + xmargin])
        ax.set_ylim([0., ymax * 1.1])
        ax.set_xlabel(label)
        ax.set_ylabel(r'\textbf{density}')

        ax.yaxis.grid(color='w', linestyle='-', zorder=2)

        self.figures.append(fig)

    def add_network(self, graph, styles={}, sym=True, labels=False):
        """
        submit eg vertex color values via styles={"vertex_color":values}

        :param graph:
        :param styles:
        :param sym:
        :return:
        """
        from igraph import Graph #, plot, GradientPalette, rescale
        assert isinstance(graph, Graph)

        if sym:
            cmap = pyplot.get_cmap("sym")
        else:
            cmap = pyplot.get_cmap("linear")

        visual_style = dict(
            edge_color='#8e908f',
            edge_width=self.params["axes.linewidth"],
            #edge_curved=0.1,
            #palette=GradientPalette('#009fda', '#e37222', 10), # for igraph
            vertex_size=100,
            vertex_color='#8e908f',
            vertex_label=range(graph.vcount())
        )

        if hasattr(graph.vs,"lat") and hasattr(graph.vs,"lon"):
            visual_style["layout"] = zip(graph.vs["lat"], graph.vs["lon"])
        elif hasattr(graph.vs,"x") and hasattr(graph.vs,"y"):
            visual_style["layout"] = zip(graph.vs["x"], graph.vs["y"])
        else:
            visual_style["layout"] = graph.layout_auto()
            print "Assign random layout for plotting."

        if styles:
            visual_style.update(styles)

        #plot(graph, target="test.pdf", **visual_style)

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()
        ax.axis("off")

        for e in graph.get_edgelist():
            edge = np.vstack((visual_style["layout"][e[0]], visual_style["layout"][e[1]]))
            ax.plot(edge[:, 0], edge[:, 1],
                    color=visual_style["edge_color"],
                    linestyle='-',
                    lw=visual_style["edge_width"],
                    zorder=1)

        x, y = zip(*visual_style["layout"])

        margin = max(0.05 * (np.max(x) - np.min(x)), 0.05 * (np.max(y) - np.min(y)))
        ax.set_xlim([np.min(x) - margin, np.max(x) + margin])
        ax.set_ylim([np.min(y) - margin, np.max(y) + margin])

        nodes = ax.scatter(x, y,
                           c=visual_style["vertex_color"],
                           s=visual_style["vertex_size"],
                           cmap=cmap,
                           vmin=np.floor(np.min(visual_style["vertex_color"])),
                           vmax=np.ceil(np.max(visual_style["vertex_color"])),
                           edgecolor='w',
                           zorder=2)

        if labels:
            for i in xrange(graph.vcount()):
                pyplot.annotate(str(i), xy=(x[i], y[i]), xytext=(3, 3), textcoords='offset points',
                                size=0.5*self.params["font.size"],
                                horizontalalignment='left', verticalalignment='bottom')

        cb = fig.colorbar(nodes, orientation='horizontal', shrink=0.66, format=r"%.1f")
        [t.set_fontsize(self.params["legend.fontsize"]) for t in cb.ax.get_xticklabels()]

        self.figures.append(fig)


    def save(self, fnames):
        assert len(fnames) == len(self.figures)
        for i, fig in enumerate(self.figures):
            fig.savefig(filename=fnames[i] + '.' + self.params['savefig.format'], bbox_inches='tight')

    def show(self):
        pyplot.show()

    def show_params(self):
        for k in matplotlib.rcParams.keys():
            print k

    def show_cmap(self, cmap_name):
        assert isinstance(cmap_name, str)
        a=np.outer(np.arange(0,1,0.01), np.ones(10))
        pyplot.figure(figsize=(2,10))
        pyplot.axis("off")
        pyplot.imshow(a,aspect='auto',interpolation='nearest', cmap=pyplot.get_cmap(cmap_name))
        pyplot.title(cmap_name,fontsize=15)
        pyplot.show()

    def update_params(self, dic):
        assert all([key in matplotlib.rcParams.keys() for key in dic.keys()])
        matplotlib.rcParams.update(dic)

    def portrait(self):
        canvas = self.params['figure.figsize']
        if canvas[1] > canvas[0]:
            raise Warning("Figure is already in portrait orientation.")
        else:
            self.params['figure.figsize'] = canvas[::-1]

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

if __name__ == "__main__":
    p = AwesomePlot.paper()

    p.show_cmap(p.dfcmp.name)

    labels = [r'$\phi$']

    x = np.arange(10)
    z = 1 - 2. * np.random.random([10, 10])
    #p.add_contour(x, x, z, sym=True)

    import igraph as ig
    g = ig.Graph.GRG(20, 0.4)
    p.add_network(g, styles={"vertex_color":np.random.random(20)})

    p.save(["test"])

    p.show()


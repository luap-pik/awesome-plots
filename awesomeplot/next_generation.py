#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Jul 07, 2016"
__version__ = "v2.0"

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

# pandas is still optional but will become standard in the next version
import pandas

from core import Plot

class AwesomePlot(Plot):
    """
    AwesomePlot class.

    The class is consistently sets reasonable matplotlib rc parameters for three different use cases.
    It can be instantiated either to create images for

    a) publications (A4, pdf)
    b) talks (A5, png)
    c) posters (not implemented)

    Images are landscape per default.
    """

    def __init__(self, output='paper', rc_spec={}, font_scale=1.1, use_pandas=False):
        super(AwesomePlot, self).__init__(output, rc_spec, font_scale)
        self.use_pandas = use_pandas
        seaborn.set_style(style="white", rc=self.rc)


    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    def add_lineplot(self, *args, **kwargs):
        if self.use_pandas:
            return self.__lineplotPD(*args, **kwargs)
        else:
            return super(AwesomePlot, self).add_lineplot(*args, **kwargs)

    def add_scatterplot(self, *args, **kwargs):
        if self.use_pandas:
            return self.__scatterplotPD(*args, **kwargs)
        else:
            return super(AwesomePlot, self).add_scatterplot(*args, **kwargs)

    def add_hist(self, *args, **kwargs):
        if self.use_pandas:
            return self.__histplotPD(*args, **kwargs)
        else:
            return super(AwesomePlot, self).add_hist(*args, **kwargs)

    @staticmethod
    def df_to_dict(df):
        assert isinstance(df, pandas.DataFrame)

        # create dict from dataframe
        dd = df.to_dict()

        # remove sub-dictionaries
        return {key: dd[key].values() for key in dd.keys()}

    ###############################################################################
    # ##                      PRIVATE FUNCTIONS                                ## #
    ###############################################################################


    def __lineplotPD(self, df, firstcol=False, legend=True, grid=True, logx=False, logy=False, loglog=False):
        assert isinstance(df, pandas.DataFrame)

        # transfer x-values to dataframe index
        if firstcol:
            df.index = df[df.columns[0]]
            df = df[df.columns[1:]]


        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        df.plot(
            kind="line",
            use_index=True,
            marker='o',
            mew=2,
            mec='w',
            colormap=self.dfcmp,
            grid=grid,
            legend=legend,
            ax=ax,
            loglog=loglog,
            logx=logx,
            logy=logy
        )

        self.figures.append(fig)

        return fig

    def __scatterplotPD(self, df, x, y, factor=None, bins=20, kind="scatter", kdeplot=False, c_map="linear"):
        assert isinstance(x, basestring)
        assert isinstance(y, basestring)

        if kdeplot:
            c = "k"
        else:
            if factor is None:
                c = self.pik_colours.colors[0]
            else:
                assert isinstance(factor, basestring)
                c = df[factor]

        xmin = df[x].min()
        xmax = df[x].max()
        ymin = df[y].min()
        ymax = df[y].max()

        settings = {
            "joint_kws": dict(alpha=1, c=c, cmap=pyplot.get_cmap(c_map)),
            "marginal_kws": dict(bins=bins, rug=False),
            "annot_kws": dict(stat=r"r", frameon=True, loc=0, handlelength=0),
            "space": 0.1,
            "kind": kind,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax)
        }

        try:
            scatter = seaborn.jointplot(x, y, data=df, **settings)
        except:
            # some kws are not valid in certain plot kinds
            pyplot.close()
            settings = {
                "annot_kws": dict(stat=r"r", frameon=True, loc=0, handlelength=0),
                "space": 0.1,
                "kind": kind,
                "xlim": (xmin, xmax),
                "ylim": (ymin, ymax)
            }
            scatter = seaborn.jointplot(x, y, data=df, **settings)

        if kdeplot:
            scatter.plot_joint(seaborn.kdeplot, shade=True, cut=5, zorder=0, n_levels=6, cmap=pyplot.get_cmap(c_map))

        fig = scatter.fig
        pyplot.close() # close JointGrid object

        self.figures.append(fig)

        return fig

    def __histplotPD(self, df, columns=None, normed=True, nbins=20, log=False, c_map="pik"):

        if columns:
            df = df[columns]


        settings = {
            "stacked": True,
            #"alpha": 0.75,
            "bins": nbins,
            "normed": normed,
            "log": log,
            "cmap": c_map
        }

        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        df.plot.hist(ax=ax, **settings)

        self.figures.append(fig)

        return fig


def test_case():
    p = AwesomePlot.talk()
    assert isinstance(p, AwesomePlot)

    label = [r"$S_B$", "S"]

    n = 20

    x = np.arange(n)
    y = np.sin(x)
    z = 1 - 2. * np.random.random([n, n])
    u = np.random.random([n, 3])

    df = pandas.DataFrame(
        data={"x": x, "y": y, "y2": -y ** 3, "y3": z[0], "y4": z[1]},
        index=x
    )

    p.add_distplot(x=x, y=z)

    p.add_contour(x=x, y=x, z=z, sym=True)

    p.add_scatterplot(y, u[:, 1]**2, labels=label, kdeplot=True)

    p.add_lineplot(x=x, lines=p.df_to_dict(df[["y", "y2"]]))

    p.add_hist(data=zip(*u))

    import networkx as nx
    G = nx.erdos_renyi_graph(n, 0.1)
    A = nx.to_scipy_sparse_matrix(G, format="dok")
    layout = np.vstack(nx.fruchterman_reingold_layout(G, center=10).values())
    layout = np.c_[layout, y]

    p.add_network(adjacency=A,
                  styles={"vertex_color": np.random.random(n),
                              "layout": layout},
                  height=True,
                  sym=False)

    p.save(["test/o" + str(i) for i in range(len(p.figures))])

    p.show()

def test_pandas():
    p = AwesomePlot.talk()
    assert isinstance(p, AwesomePlot)

    p.use_pandas = True

    p.set_default_colours("pik")

    n = 200

    x = np.arange(n)
    y = np.sin(x)
    z = 1 - 2. * np.random.random([n, n])

    u = np.random.random([n, 3])

    df = pandas.DataFrame(
        data={"x": x, "y": y, "y2": -y**3,"y3": z[0], "y4": z[1]},
        index=x
    )

    assert isinstance(df, pandas.DataFrame)

    p.add_lineplot(df, firstcol=True)

    p.add_scatterplot(df, "y", "y4", factor="x", c_map="discrete")

    p.add_hist(df, c_map="discrete")

    p.save(["test/p" + str(i) for i in range(len(p.figures))])

    p.show()

if __name__ == "__main__":
    test_case()
    test_pandas()



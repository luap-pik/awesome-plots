#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Jul 07, 2016"
__version__ = "v1.2"

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
        """
            Initialise an instance of Plot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """

        assert output in ["paper", "talk", "poster", "notebook"]

        self.set_default_colours('pik')

        rc = {'xtick.direction': 'in',
              'ytick.direction': 'in',
              'verbose.level': 'helpful',
              'lines.linewidth': 3,
              'axes.linewidth': 3
              }

        if rc:
            rc.update(rc_spec)

        seaborn.set_context(output, font_scale=font_scale, rc=rc)
        seaborn.set_style(style="white", rc=rc)

        # default behaviour
        self.use_pandas = use_pandas

        self.figures = []

    @classmethod
    def paper(cls):
        """
        Class method yielding an Plot instance of type "paper"

        Parameters
        ----------
        cls: object
            Plot class

        Returns
        -------
        instance of class Plot

        """

        rc = dict()
        rc['figure.figsize'] = (11.69, 8.268)  # A4
        rc['savefig.format'] = 'pdf'
        rc['pdf.compression'] = 6  # 0 to 9
        rc['pdf.fonttype'] = 42
        rc['savefig.dpi'] = 300

        return cls(output='paper', rc_spec=rc)

    @classmethod
    def talk(cls):
        """
        Class method yielding an Plot instance of type "talk"

        Parameters
        ----------
        cls: object
            Plot class

        Returns
        -------
        instance of class Plot

        """
        rc = dict()
        rc['figure.figsize'] = (8.268, 5.872)  # A5
        rc['savefig.format'] = 'png'
        rc['savefig.dpi'] = 300

        return cls(output='talk', rc_spec=rc)

    @classmethod
    def poster(cls):
        """
        Class method yielding an Plot instance of type "icon"

        Parameters
        ----------
        cls: object
            Plot class

        Returns
        -------
        instance of class Plot

        """
        return cls(output='poster')

    @classmethod
    def notebook(cls):
        """
        Class method yielding an Plot instance of type "paper"

        Parameters
        ----------
        cls: object
            Plot class

        Returns
        -------
        instance of class Plot

        """
        return cls(output='notebook')

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    def add_lineplot(self, **kwargs):
        if self.use_pandas:
            return self.__lineplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_lineplot(**kwargs)

    def add_distplot(self, **kwargs):
        if self.use_pandas:
            return self.__distplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_distplot(**kwargs)

    def add_contour(self, **kwargs):
        if self.use_pandas:
            return self.__contourplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_contour(**kwargs)

    def add_scatterplot(self, **kwargs):
        if self.use_pandas:
            return self.__scatterplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_scatterplot(**kwargs)

    def add_hist(self, **kwargs):
        if self.use_pandas:
            return self.__histplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_hist(**kwargs)

    def add_network(self, **kwargs):
        if self.use_pandas:
            return self.__networkplotPD(**kwargs)
        else:
            return super(AwesomePlot, self).add_network(**kwargs)

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


    ###############################################################################
    # ##                      PRIVATE FUNCTIONS                                ## #
    ###############################################################################


    def __lineplotPD(self):
        print 42
        pass

    def __distplotPD(self):
        pass

    def __contourplotPD(self):
        pass

    def __scatterplotPD(self):
        pass

    def __histplotPD(self):
        pass

    def __networkplotPD(self):
        pass

def test_case():
    p = AwesomePlot.talk()
    assert isinstance(p, AwesomePlot)

    # p.show_cmap(p.dfcmp.name)

    labels = [r'$\phi$']

    n = 20

    x = np.arange(n)
    y = np.sin(x)
    z = 1 - 2. * np.random.random([n, n])

    u = np.random.random([n, 3])

    p.add_hist(data=zip(*u))

    p.add_lineplot(x=x, lines={"y": y})

    p.add_scatterplot(x={0: x}, y={0: y})

    p.add_distplot(x=x, y=z)

    p.add_contour(x=x, y=x, z=z)

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

    df = pandas.DataFrame(data={"y": np.linspace(0, 10, 10) ** 2,
                                "y2": np.linspace(0, 10, 10) ** 3,
                                "y3": np.linspace(0, 10, 10) ** 2.1,
                                "y4": np.linspace(0, 10, 10) ** 2.3,
                                "y5": np.linspace(0, 10, 10) ** 2.5},
                          index=np.linspace(0, 10, 10))

    df.plot(kind="line",
            use_index=True,
            marker='o',
            mew='3',
            mec='w',
            colormap="discrete",
            grid=True)
            #title=title,
            # legend=None,
            #ax=axes)

    p.show()

if __name__ == "__main__":
    test_case()



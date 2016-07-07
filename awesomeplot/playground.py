#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Jul 07, 2016"
__version__ = "v1.2"

"""
Module contains class AwesomePlot.

Derivative of the matplotlib module. The aim is to create
visually attractive, unambigous and colour-blind-friendly
images, customised for PIK corporate design.

In this version, data is handled using pandas and styles are
defined using seaborn.

"""

# Import NumPy for the array object and fast numerics.
import numpy as np

# import matplotlib and submodules
import matplotlib
from matplotlib import pyplot
from matplotlib import cycler
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color

import pandas

import seaborn

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


    def __init__(self, output='paper', rc_spec=None, font_scale=2):
        """
            Initialise an instance of AwesomePlot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """
        assert output in ["paper", "talk", "poster", "notebook"]

        rc = {"image.cmap": "pik",
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              }

        if rc:
            rc.update(rc_spec)

        seaborn.set_context(output, font_scale=font_scale, rc=rc)
        seaborn.set_style(style="ticks", rc=rc)

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
        Class method yielding an AwesomePlot instance of type "talk"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """
        rc = dict()
        rc['figure.figsize'] = (8.268, 5.872)  # A5
        rc['savefig.format'] = 'png'
        rc['savefig.dpi'] = 300

        return cls(output='talk', rc_spec=rc)

    @classmethod
    def poster(cls):
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
        return cls(output='poster')

    @classmethod
    def notebook(cls):
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
        return cls(output='notebook')

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    def add_lineplot(self, df, labels=['x', 'y'], title=None):


        assert len(labels) == 2
        assert isinstance(df, pandas.DataFrame)

        fig, axes = pyplot.subplots(nrows=1, ncols=1)



        lines = df.plot(kind="line",
                        use_index=True,
                        marker='o',
                        mew='3',
                        mec='w',
                        colormap="discrete",
                        grid=True,
                        title=title,
                        #legend=None,
                        ax=axes)


        #axes.fill_between()


        self.figures.append(fig)

    def show(self):
        pyplot.show()



if __name__ == "__main__":
    p = AwesomePlot.talk()
    assert isinstance(p, AwesomePlot)

    df = pandas.DataFrame(data={"y": np.linspace(0, 10, 10)**2,
                                "y2": np.linspace(0, 10, 10)**3,
                                "y3": np.linspace(0, 10, 10) **2.1,
                                "y4": np.linspace(0, 10, 10) ** 2.3,
                                "y5": np.linspace(0, 10, 10) ** 2.5},
                          index=np.linspace(0, 10, 10))



    p.add_lineplot(df)


    # p.save(["test"])

    p.show()


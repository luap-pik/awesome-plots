import networkx as nx
import numpy as np
import pandas

def test_case():
    from awesomeplot.core import Plot

    p = Plot.talk(font_scale=1.2)
    assert isinstance(p, Plot)


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

    p.add_lineplot(x=x, lines={"y":df.y, "y2": df.y2})

    p.add_hist(data={i: u[i] for i in range(u.shape[1])})

    import networkx as nx
    pos = np.random.random([100, 2])

    g = nx.random_geometric_graph(100, 0.11, dim=3, pos={i: pos[i, :] for i in range(100)})
    A = nx.to_scipy_sparse_matrix(g, format="dok")

    z = np.zeros(g.number_of_nodes())
    z[np.array(g.degree().values()) > 2] = 1

    layout = np.c_[pos, z]

    p.add_network(A, height=True, styles={"vertex_color": np.random.random(100), "layout": layout})

    p.show()

    p.save(["test/o" + str(i) for i in range(len(p.figures))])



def test_pandas():
    from awesomeplot.core import PandasPlot

    p = PandasPlot.talk()
    assert isinstance(p, PandasPlot)

    p.set_default_colours("pik")

    n = 100

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

    p.add_scatterplot(df, "y", "y4", factor="x", kdeplot=True)

    p.add_hist(df, c_map="discrete")

    p.save(["test/p" + str(i) for i in range(len(p.figures))])

    p.show()

if __name__ == "__main__":
    import os
    if not os.path.exists("test"):
        os.mkdir("test")
    test_case()
    #test_pandas()





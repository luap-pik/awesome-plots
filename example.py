import networkx as nx
import numpy as np

from awesomeplot.pandas import AwesomePlot

p = AwesomePlot.paper()

assert isinstance(p, AwesomePlot)

pos = np.random.random([100, 2])

g = nx.random_geometric_graph(100, 0.11, dim=3, pos={i: pos[i, :] for i in range(100)})
A = nx.to_scipy_sparse_matrix(g, format="dok")

z = np.zeros(g.number_of_nodes())
z[np.array(g.degree().values()) > 2] = 1

layout = np.c_[pos, z]

p.add_network(A, height=True, styles={"vertex_color": np.random.random(100), "layout": layout})

p.save(["test"])

p.show()

quit()

# p.show_params()

x = np.arange(10)
z = 1 - 2. * np.random.random([10, 10])
#p.add_contour(x, x, z, sym=True)

p.set_default_colours("discrete")
p.add_lineplot(x, {0: z})

p.show()






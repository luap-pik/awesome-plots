import igraph as ig
import numpy as np
from awesomeplot.core import AwesomePlot


p = AwesomePlot.paper()


# p.show_params()

labels = [r'$\phi$']

x = np.arange(10)
z = 1 - 2. * np.random.random([10, 10])
#p.add_contour(x, x, z, sym=True)


g = ig.Graph.GRG(20, 0.4)
p.add_network(g, styles={"vertex_color":np.random.random(20)})

p.save(["test"])

p.show()
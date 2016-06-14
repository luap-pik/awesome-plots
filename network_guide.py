from igraph import plot, palettes, Graph
import numpy as np

edgelist = [(0, 1), (1, 2), (2, 1), (2, 0)]

graph =  Graph(n=3, edges=edgelist, directed=True)
assert isinstance(graph, Graph)

# assign random "Iota" for each edge
iota = np.random.random(3)

visual_style = dict(
    margin=30,
    edge_color="orange",
    edge_curved=0.1,
    edge_width= 5,
    edge_label="",
    edge_label_dist=1,
    layout=graph.layout_auto(),
    palette=palettes["heat"],
    vertex_size=30,
    vertex_shape="circle",
    vertex_color=256*iota,
    vertex_frame_color="white",
    vertex_frame_width=3,
    vertex_label=range(graph.vcount()),
    vertex_label_dist=0,
    vertex_label_color="black",
    vertex_label_size=12,
    vertex_label_angle=0
)

plot(graph, "test_plot.pdf", **visual_style)
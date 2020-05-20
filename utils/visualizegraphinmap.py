import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import adj2edges

def draw_node_edge(adj, coords, savepath):
    '''
    Draw the nodes and the edges
    '''
    
    fig, ax = plt.subplots()
    plt.figure(1)
    for (x, y, heading) in coords:
        plt.scatter(x,y, marker='o', c="#ff1212", s=0.5)
    
    edges = adj2edges(adj)
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]
        x1 = coords[node1][0]
        y1 = coords[node1][1]
        x2 = coords[node2][0]
        y2 = coords[node2][1]

        plt.plot([x1, x2], [y1, y2], color='g', alpha=1, linewidth=0.01)

    fig.savefig(os.path.join(savepath, 'graph.png'),dpi=600)#,format='eps')


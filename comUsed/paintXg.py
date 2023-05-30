import matplotlib.pyplot as plt
import networkx as nx

def drawSca(list, color='blue', size=1, isTag = False):
    data_x = [x[0] for x in list]
    data_y = [x[1] for x in list]
    plt.scatter(x=data_x, y=data_y, color=color, s=size)
    if isTag:
        for i in range(len(list)):
            plt.text(data_x[i], data_y[i], i + 1, fontsize=8, color="r", style="italic", weight="light", verticalalignment='center',
                     horizontalalignment='right')

def drawLine(list, color='blue'):
    data_x = [x[0] for x in list]
    data_y = [x[1] for x in list]
    plt.plot(data_x, data_y, color=color)

def drawNetWork(nodes, links):
    G = nx.Graph()
    for link in links:
        G.add_edge(link[0], link[1])
    pos = {}
    for node in nodes:
        pos[node['id']] = (node['x'], node['y'])
    # explicitly set positions
    options = {
        "font_size": 0.5,
        "node_size": 1,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    nx.draw_networkx(G, pos, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
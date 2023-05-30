import pandas as pd
import networkx as nx
from temp import *
# 与网络计算有关的方法 节点数组[node1, node2, node3, node4]

class Node():
    def __init__(self, id, x, y, degree):
        self.y = y
        self.x = x
        self.id = id
        self.degree = degree

class Edge():
    def __init__(self, staId, endId):
        self.staId, self.endId = staId, endId

# 获得一个基础路网
def getNet():
    roads = json_load(r'D:\pycharmProject\logic\dataSet\road_sta_end.json')
    edges, nodes = [], []
    netWork = {}
    for road in roads:
        staId, endId = road[0][2], road[1][2]
        staLat, staLng, endLat, endLng = road[0][0], road[0][1], road[1][0], road[1][1]
        edges.append(Edge(staId, endId))
        # edges.append([staId, endId])
        idL = []
        if staId not in idL:
            nodes.append(Node(id=staId, x=staLat, y=staLng, degree=''))
            idL.append(staId)
        if endId not in idL:
            nodes.append(Node(id=endId, x=endLat, y=endLng, degree=''))
            idL.append(endId)
        edges.append(Edge(staId=staId, endId=endId))
    # 生成 edges和nodes
    return nodes, edges


# 计算网络节点的度
def getNet_nodality(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        G.add_edge(edge.staId, edge.endId)
    dic_degree = G.degree
    return dic_degree


# 计算网络节点的介数
def getNet_betweeness(nodes, edges):
    # 点的介数
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        G.add_edge(edge.staId, edge.endId)
    return nx.edge_betweenness(G), nx.betweenness_centrality(G)


# 计算网络平均效率
def getNet_aveEffciency(nodes, edges):
    G = nx.Graph()
    idL = []
    n = len(nodes)
    for node in nodes:
        G.add_node(node.id)
        idL.append(node.id)
    for edge in edges:
        G.add_edge(edge.staId, edge.endId)
    shortDis = []
    for i in idL:
        for j in idL:
            if i!=j:
                try:
                    shortDis.append(len(nx.shortest_path(G, source=i, target=j))-1)
                except:
                    print(i, j, "这两个节点不存在边")

    shortDis_inverse = [1/x for x in shortDis]
    netEffcie = sum(shortDis_inverse) / n / (n-1)
    return netEffcie




if __name__ == '__main__':

    nodes, edges = getNet()
    ans = getNet_aveEffciency(nodes, edges)


    pass
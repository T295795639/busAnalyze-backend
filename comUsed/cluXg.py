# 聚类算法相关
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  # python自带的迭代器模块
from collections import Counter, defaultdict
from temp import *
from comUsed import *

f = r"D:\pycharmProject\logic\tempData"

# 层次聚类
def hieCluster(X, n_clusters_, plot):
    '''
    :param X: 需要聚类的点,放到一个array中
    :param n_clusters_: 需要得到的聚类个数
    :return: 聚类lable标签
    '''
    # 设置分层聚类函数
    linkages = ['ward','average','complete']
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters = n_clusters_)
    # 训练数据
    ac.fit(X)
    # 每个数据的分类
    lables = ac.labels_
    if plot == True:
        plt.figure(1)  # 绘图
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            # 根据lables中的值是否等于k,重新组成一个True、False的数组
            my_members = lables == k
            # X[my_members,0]取出my_members对应位置为True的值的横坐标
            plt.plot(X[my_members,0],X[my_members,1],col+'.')
        plt.title('Estimated number of clusters:%d' % n_clusters_)
        plt.show()
    else:
        pass
    return lables

def getStation_hieClu(k):
    '''
    :param k: 聚类个数
    :return:站点聚类结果 带标签
    '''
    k = int(k)
    stations = json_load(r'D:\pycharmProject\logic\dataSet\staIdL.json')
    # stations = list(filter(lambda x: x[2]>10000, stations))
    data = np.array([x[0:2] for x in stations])
    # print('聚类要用的数据', data[0:10])
    lables = hieCluster(np.array(data), k, False)
    # 将标签数量小于10的标签置-1
    count = Counter(lables)
    L = []
    # 是否删将过少散点都删了 并且存入'-1'
    # 将数量小于10的lable存起来
    # for x in range(k):
    #     if count[x] <= 10:
    #         L.append(x)
    # for x in range(len(lables)):
    #     if lables[x] in L:
    #         lables[x] = -1
    # 给数据加标签
    dataWithLable = []
    for d, lable in zip(data, lables):
        d = list(d)
        d.append(int(lable))
        dataWithLable.append(d)
    # 加线路名 加站点编号
    # data = list(map(lambda x, y: x+[y[4], y[3], y[-1]], data, List))
    return dataWithLable


# 计算质心
def getCluPoint():
    '''
    使用聚类结果计算质心
    :return: 质心字典
    '''
    hieCluData = json_load(r'D:\pycharmProject\logic\4.站点聚类\hieClu.json')
    cluAns = defaultdict(list)
    for station in hieCluData:
        cluId = int(station[-1])
        cluAns[cluId].append(station)
    # json_dump(cluAns, 'outData/cluAns.json')
    cluPoint = {}
    for key, value in cluAns.items():
        cluPoint[key] = gpsXg.center_geolocation(value)
    pointL = list(cluPoint.values())
    return cluPoint


# 计算聚类的连接关系
def getCluEdge():
    '''
    得到聚类后的边
    :return:
    '''


if __name__ == '__main__':
    cluPoint = getCluPoint()
    idList = list(cluPoint.keys())
    valueL = list(cluPoint.values())
    ans = []
    for id, value in zip(idList, valueL):
        value.append(id)
        ans.append(value)

    json_dump(ans, f+'\\cluPointL.json')
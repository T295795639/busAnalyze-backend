import sqlite3
import os
import pandas as pd
from geopy.distance import geodesic
import json
from collections import defaultdict, Counter
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle  # python自带的迭代器模块
from sklearn.cluster import AgglomerativeClustering

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def getdis_geo(pos1, pos2):
    return geodesic((pos1[1], pos1[0]), (pos2[1], pos2[0])).m


def json_load(filename):
    return json.load(open(filename, encoding='utf-8'))


def json_dump(data, tof):
    return json.dump(data, open(tof, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def staNameWash(df, k):
    '''
    清洗同名站点
    :param df: 站点dataframe
    Index(['内码', '线路', '方向', '站序', '是否大站', '限速值(km/h)', '站名', '经度', '纬度', '站点ID',
       '距离', '线路号', 'lineNo'])
    :param k: 每一个簇的收敛阈值
    :return: [[[松花江站，站点id，经度，纬度]， [松花江站，站点id，经度，纬度]],[],[],[],[]]
    '''
    grouped = df.groupby('站名')
    ansList = []
    linkDic = defaultdict(list)
    for staName, group in grouped:
        # 对每一坨站点
        List = []
        for index, row in group.iterrows():
            if row['纬度'] <= 90 and row['纬度'] >= -90:
                # 去除该站点的同名数据
                sta_temp = [row['站名'], row['站点ID'], row['经度'], row['纬度'], row['线路号']]
                staId_temp = sta_temp[1]
                # 给站点去重 小于1km的不再加入
                # 首站点加入
                if index == 0:
                    List.append(sta_temp)
                else:
                    # 判断是否该加进去
                    F = True
                    for sta in List:
                        # print((sta_temp[2], sta_temp[3]), (sta[2], sta[3]))
                        if getdis_geo((sta_temp[2], sta_temp[3]), (sta[2], sta[3])) < k:
                            linkDic[sta[1]].append(sta_temp[1])
                            F = False
                    if F:
                        List.append(sta_temp)
        ansList.append(List)
    return linkDic, ansList

def staId2matrixIndex(sta_info, header):
    staName, staId = sta_info[0], sta_info[1]
    lng, lat = sta_info[2], sta_info[3]
    # 使用名字去重后的station
    df_station = pd.read_csv(r'E:\buStation_nanChang_netWork\output\station_dropDup_2\station.csv')
    if staId in header:
        return header.index(staId)
    else:
        df_findByStaName = df_station.loc[df_station['站名'] == staName, :]
        for index, row in df_findByStaName.iterrows():
            if getdis_geo((lng, lat), (row['经度'], row['纬度'])) < 1000:
                return header.index(row['站点ID'])

def getLineGps(lineNoL, df_gps, tof):
    '''
    找线路的一条gps路线
    :param lineNoL: 线路数组
    :param df_gps: 一块gps
    :param tof: 存在哪里
    :return:
    '''
    # 每条线路有几个carNo
    dic_line_carNo = {}
    for line in lineNoL:
        df = df_gps.loc[df_gps['lineNo'] == line, :]
        carNoL = df['busNo'].unique().tolist()
        dic_line_carNo[line] = carNoL
    print(dic_line_carNo)
    # 提取每条路的上下行
    df_gps.sort_values(by='time', inplace=True)

    df_up = df_down = pd.DataFrame()
    erroLine = []
    for line in lineNoL:
        carNoL = dic_line_carNo[line]
        F_up, F_down = True, True
        # 搜索该line的上行gps和下行gps
        # 上行是3 00000 4  下行是4 11111 3
        for carNo in carNoL:
            df = df_gps.loc[((df_gps['lineNo'] == line) & (df_gps['busNo'] == carNo)), :]
            l, i = len(df), 0
            while i < l-40:
                # print(i)
                # 找上行
                if df.iloc[i, :]['isUpDown'] == 3 and df.iloc[i + 1, :]['isUpDown'] == 0 and df.iloc[i + 40, :]['isUpDown'] == 0:
                    for j in range(i + 40, l):
                        if df.iloc[j, :]['isUpDown'] == 4:
                            # F_up = False
                            df_up = df.iloc[i:j, :]
                            i = j
                            df_up.to_csv(tof + '\\' + line + '_up_' + carNo + '_' + df.iloc[i, :]['time'], index=False)
                            break
                # 找下行
                if df.iloc[i, :]['isUpDown'] == 4 and df.iloc[i + 1, :]['isUpDown'] == 1 and df.iloc[i + 40, :]['isUpDown'] == 1:
                    for j in range(i + 40, l):
                        if df.iloc[j, :]['isUpDown'] == 3:
                            # F_down = False
                            df_down = df.iloc[i:j, :]
                            i = j
                            df_up.to_csv(tof + '\\' + line + '_up_' + carNo + '_' + df.iloc[i, :]['time'], index=False)
                            break
            # 打印条件
            if F_up == False and F_down == False:
                tof1, tof2 = tof + '\\' + line + '_up.csv', tof + '\\' + line + '_down.csv'
                df_up.to_csv(tof1, index=False)
                df_down.to_csv(tof2, index=False)
                break

    #     if F_up == True or F_down == True:
    #         print(line, 'Erro')
    #         erroLine.append(line)
    #     else:
    #         print(line, 'Sucess')
    #
    # json_dump(erroLine, tof + '\\' + 'erroLine.json')


def getSta2StaGps(df, beginSta, endSta):
    '''
    得到某两个站点之间的轨迹点
    :param df: 轨迹点DataFrame data/gps_(2021-09-09 12-14)
    :param beginSta: 开始站点ID
    :param endSta: 结束站点ID
    :return: 这两个站点间的轨迹点DataFrame
    '''
    # 站点表
    df_station = pd.read_csv(r'E:\buStation_nanChang_netWork\output\1_lineNo_clearOut\station.csv')
    # 获取线路号
    lineNo = df_station.loc[df_station['站点ID'] == beginSta, ['lineNo']].values[0][0]
    # 提取该线路的轨迹点
    gps_lineNo = df.loc[df['lineNo'] == lineNo, :].sort_values(by=['time'])
    # 站点位置
    beginPos = df_station.loc[df_station['站点ID'] == beginSta, ['经度', '纬度']].values[0]
    endPos = df_station.loc[df_station['站点ID'] == endSta, ['经度', '纬度']].values[0]

    gps_lineNo.to_csv('tagStation.csv')
    print(beginPos, endPos, gps_lineNo)


def list2Features(List):
    '''
    staList转地图要素
    :param staList: [[站点名，站ID，经度，纬度]...]
    :return: 要素
    '''
    features = []
    for x in List:
        for y in x:
            y[0] = y[0].replace(' ', '')
            features.append({
                'type': 'feature',
                'properties': {
                    '站名': y[0],
                    '站点ID': y[1]
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [y[2], y[3]]
                }
            })

# input一个graph和起始点s
def BFS(graph, s):
    ansL = []
    # 创建队列
    queue = []
    # 将起始点s放入队列，假设起始点为‘A’
    queue.append(s)
    # set():创建一个无序不重复元素集，可进行关系测试，删除重复数据,还可以计算交集、差集、并集
    seen = set()
    # 'A'我们已经见过，放入seen
    seen.add(s)
    # 当队列不是空的时候
    while len(queue) > 0:
        # 将队列的第一个元素读出来，即‘A’
        vertex = queue.pop(0)
        # graph['A']就是A的相邻点：['B','C'],将其储存到nodes
        nodes = graph[vertex]
        # 遍历nodes中的元素，即['B','C']
        for w in nodes:
            # 如果w没见过
            if w not in seen:
                queue.append(w)
                # 加入seen表示w我们看见过了
                seen.add(w)
        ansL.append(vertex)
    return ansL

def BFS_test():
    # 测试BFS
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D', 'E'],
        'D': ['B', 'C', 'E', 'F'],
        'E': ['C', 'D'],
        'F': ['D']
    }
    ans = BFS(graph, 'A')
    print(ans)

def roadGpSplit(gpsList, station):
    pass

def drawSca(list, color='blue', size=1):
    data_x = [x[0] for x in list]
    data_y = [x[1] for x in list]
    plt.scatter(x=data_x, y=data_y, color=color, s=size)
    # for i in range(len(list)):
    #     plt.text(data_x[i], data_y[i], i + 1, fontsize=8, color="r", style="italic", weight="light", verticalalignment='center',
    #              horizontalalignment='right')

def drawLine(list, color='blue'):
    data_x = [x[0] for x in list]
    data_y = [x[1] for x in list]
    plt.plot(data_x, data_y, color=color)

def getLength(pathL):
    L = 0
    for i in range(len(pathL)-1):
        L = L + getdis_geo(pos1=pathL[i], pos2=pathL[i+1])
    return L

def getNei2Point(pointL, A):
    minIndex = -999
    minDis = 9e10
    minIndex2 = -999
    minDis2 = 9e10
    for i, point in enumerate(pointL):
        dis = getdis_geo(point, A)
        # 是第一小的点吗？
        if dis < minDis:
            minDis, minIndex = dis, i
        elif dis < minDis2:
            # 虽然不是第一小的点？ 是第二小的点吗？
            minDis2, minIndex2 = dis, i
    return (minIndex, pointL[minIndex]), (minIndex2, pointL[minIndex2])

def sta2Gps(gpsList, staList, type=1):
    '''
    获取站点的插入位置
    :param gpsList: 轨迹点
    :param staList: 站点
    :return: indexlist indexlist_shift gpsList 插入前的位置 插入后的位置 轨迹点列表
    '''
    indexList = []
    # ********************************对gps点进行缩放 取位置最近的点们 o(n2) 速度极慢**************************************
    if type == 1:
        # 插入n+1个位置 选择插入后路径最短的那个
        # 插入位置获取
        insertL = list(range(0, len(gpsList)+1))
        # print(insertL)
        for sta in staList:
            # 记录每一个站点插入后的线路长度
            minIndex = -1
            minLen = 99999
            for i in insertL:
                pathL = deepcopy(gpsList)
                pathL.insert(i, sta)
                route_length = getLength(pathL)
                if route_length < minLen:
                    minLen = route_length
                    minIndex = i
            indexList.append(minIndex)
    else:
        for sta in staList:
            pointA, pointB = getNei2Point(pointL=gpsList, A=sta)
            if pointA[0] > pointB[0]:
                indexList.append(pointA[0])
            else:
                indexList.append(pointB[0])
    # 加上偏移量 得到站点插入后的位置 而不是插入位置
    indexList_shift = [x+i for i, x in enumerate(indexList)]

    for sta, insertIndex in zip(staList, indexList_shift):
        gpsList.insert(insertIndex, sta)

    return indexList, indexList_shift, gpsList

def gpSplit(gpsList, staList, type):
    '''
    生成切好的gps
    :param gpsList:
    :param staList:
    :return:
    '''
    ansL = []
    _, indexList, gpsList = sta2Gps(staList=staList, gpsList=gpsList, type=type)
    # print(indexList)
    for i in range(len(indexList)-1):
        List = gpsList[indexList[i]:indexList[i+1]+1]
        ansL.append(List)
    return ansL

def staID2Pos_convert(df_station, staId):
    pos = df_station.loc[df_station['站点ID']==staId, ['经度', "纬度"]].values[0].tolist()
    return pos

class Test():

    def sta2Gps(self, lineName='5_down', type=1):
        gpsL = json_load(r'E:\buStation_nanChang_netWork\DP\DP\{0}.json'.format(lineName))['coordinates']
        staL = json_load(r'E:\buStation_nanChang_netWork\output\2_station_dropDup\line_pos.json')[lineName]
        drawLine(gpsL)
        drawSca(gpsL, 'blue')
        drawSca(staL, 'red')
        plt.title('站点Insert之前')
        plt.show()
        _, indexList, gpsL = sta2Gps(gpsList=gpsL, staList=staL, type=type)
        drawLine(gpsL)
        drawSca(gpsL, 'blue')
        drawSca(staL, 'red')
        plt.title('站点Insert之后')
        plt.show()
        return indexList, gpsL

    def gpSplit(self, lineName='5_down'):
        gpsL = json_load(r'E:\buStation_nanChang_netWork\DP\DP\{0}.json'.format(lineName))['coordinates']
        staL = json_load(r'E:\buStation_nanChang_netWork\output\2_station_dropDup\line_pos.json')[lineName]
        gps_split = gpSplit(gpsList=gpsL, staList=staL, type=0)
        # print(gps_split)
        lenList = [len(x) for x in gps_split]
        # print(lenList)
        colorL = ['red', 'orange', 'gray', 'black', 'blue']
        for i, gpsRoad in enumerate(gps_split):
            drawLine(gpsRoad, colorL[i%5])
        plt.title('根据站点划分路段')
        plt.show()

    def getNetWork_aveDe(self):
        points = list(range(len(json_load(r'E:\buStation_nanChang_netWork\output\6_netWork_road2point\point.json'))))
        edges = json_load(r'E:\buStation_nanChang_netWork\output\6_netWork_road2point\edge.json')
        pWeights, eWeights = '', ''
        net = NetWork(points, edges, pWeights, eWeights)
        aveDe = net.getNetWork_aveDe()
        print(aveDe)

    def getLineGps(self):
        lineNoL = [2]
        df_gps = pd.read_sql(con=sqlite3.connect(r'E:\buStation_nanChang_netWork\db\nanChang.db'), sql='SELECT * FROM BUS_GPS WHERE time BETWEEN "2021-09-09 12:00:00" and "2021-09-09 12:30:00"')
        tof = r'E:\buStation_nanChang_netWork\output\4_getRoadGps\GPS'
        getLineGps(lineNoL, df_gps, tof)

# 网络算法都在这
class NetWork():

    points = edges = pWeights = eWeights = []

    def __init__(self, points, edges, pWeights, eWeights):
        '''
        初始化类
        :param points: point index 数组
        :param edges: [ponit1, point2] 数组
        :param pWeights: [pWeight1, pWeight2]
        :param eWeights: [eWeight1, eWeight2]
        '''
        self.points = points
        self.edges = edges
        self.pWights = pWeights
        self.eWeights = eWeights

    def getNetWork_aveDe(self):
        # 点的个数
        N = len(self.points)
        # 所有点的度相加
        De = len(self.edges)
        return N/De

class FeatureProduce():

    def getHeatLine(self, staPos, endPos):
        lng1, lat1 = staPos[0], staPos[1]
        lng2, lat2 = endPos[0], endPos[1]

    def point2HeatFeatures(self, pointL, weightL):
        dic = {}
        dic['features'] = []
        for point, weight in zip(pointL, weightL):
            dic['features'].append({
                "geometry": {
                    "coordinates": point,
                    "type": "Point"
                },
                "type": "Feature",
                "properties": {
                    "weight": weight
                }
            })
        return dic

    def line2HeatFeatures(selfs, lineL, weightL):
        dic = {}
        dic['features'] = []
        for line, weight in zip(lineL, weightL):
            dic['features'].append({
                "geometry": {
                    "coordinates": line,
                    "type": "LineString"
                },
                "type": "Feature",
                "properties": {
                    "weight": weight
                }
            })
        return dic

def getPath(f):
    fNameL = os.listdir(f)
    fPathL = [os.path.join(f, x) for x in fNameL]
    return fNameL, fPathL

def getGpsPoints(df):
    points = df.loc[:, ["lng", "lat", "time"]].values.tolist()
    return points

def getHawthorn(gpsPoints, stations):
    ans = []
    for point in gpsPoints:
        for staIndex, station in enumerate(stations):
            if getdis_geo(station[0:2], point[0:2]) < 50 and [station[2], point[2]] not in ans:
                ans.append([station[2], point[2]])
    return ans

def searchSta(point, stations):
    subLng = 0.0005405642952496237
    subLat = 0.0003986414159840024
    for station in stations:
        if abs(station[0]-point[0]) > subLng or abs(station[1]-point[1]) > subLat:
            continue
        if getdis_geo(point, station[0:2]) < 50:
            return station
    return False

def delZero(numList, type):
    '''
    替换0为平均值
    :param numList: 数组
    :return: []
    '''
    if type == 'fillAve':
        ave = sum(numList)/len(numList)
        for i, num in enumerate(numList):
            if num == 0:
                numList[i] = ave
    if type == 'fillMin':
        Min = min(numList)
        for i, num in enumerate(numList):
            if num == 0:
                numList[i] = Min + 1
    return numList

def delMax(numList):
    '''
    删除太大的点
    :param numList: 数组
    :return: []
    '''
    ave = sum(numList) / len(numList)
    for i, num in enumerate(numList):
        if num > ave*3:
            numList[i] = ave*3
    return numList

# *********************************************** 聚类算法 *********************************************************
# 层次聚类
def hieCluster(X, n_clusters_, plot):
    '''
    :param X: 需要聚类的点,放到一个array中
    :param n_clusters_: 需要得到的聚类个数
    :return: 聚类lable标签
    '''
    # 设置分层聚类函数
    linkages = ['ward', 'average', 'complete']
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
    # 训练数据
    ac.fit(X)
    # 每个数据的分类
    lables = ac.labels_
    if plot == True:
        # plt.figure(1)  # 绘图
        # plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            # 根据lables中的值是否等于k,重新组成一个True、False的数组
            my_members = lables == k
            # X[my_members,0]取出my_members对应位置为True的值的横坐标
            plt.plot(X[my_members,0],X[my_members,1],col+'.')
        plt.title('Estimated number of clusters:%d' % n_clusters_)
        # plt.show()
    else:
        pass
    return lables

# 通过聚类得到站点聚类数据
def getStationClu(stations=json_load('D:\pycharmProject\logic\dataSet\staIdL.json'), K=110):
    '''
    得到站点的聚类结果 [新的站点, 站点标签]
    :param stations: [sta1, sta2, sta3]
    :return: [[sta1, cluId1], [sta2, cluId2]]
    '''
    # 筛选路口
    # stations = list(filter(lambda x: x[2]>10000, stations))
    cluDate_pre = np.array([x[0:2] for x in stations])
    # print(cluDate_pre)
    lables = hieCluster(cluDate_pre, K, True)
    counted = Counter(lables)
    lables_valid = []
    for lable, count in counted.items():
        # if count>=10:
            lables_valid.append(lable)
    # 存储stations在哪个聚类中 站点id:聚类lable 还是 聚类lable:站点id 选第二种
    dic_lable2station = defaultdict(list)
    dic_station2lable = {}
    stationWithLable = []
    for lable, station in zip(lables, list(stations)):
        station.append(int(lable))
        stationWithLable.append(station)
        if lable in lables_valid:
            dic_lable2station[int(lable)].append(station[0:-1])
            dic_station2lable[station[2]] = int(lable)
    # print(stationWithLable)
    # json_dump(stationWithLable, 'hieClu.json')
    # json_dump(dic_lable2station, r'D:\pycharmProject\busRoute_nanchang\data\tempData\lable2stations.json')
    # json_dump(dic_lable2station, 'lable2stations.json')
    # json_dump(dic_station2lable, 'station2lables.json')
    # "输入:" stations   "输出:" dic_lable2station
    return dic_lable2station

# 删去一部分 再做聚类
def getStationClu_sub(stations, sub_stations, K):
    '''
    删去一部分节点的聚类效果
    :param stations: 站点列表
    :param sub_stations: [[站点簇1], [站点簇2]]
    :return: lable2stations
    '''
    dic_staId_lnglat = json_load(r'D:\pycharmProject\busAnalyze-backend\data\dic_staId_lnglat.json')
    K = K - len(sub_stations)
    # 存放删除后的站点 结果
    stations_ans = []
    # 存放需要删除的站点
    sub_stations2 = []
    for stations_sub in sub_stations:
        for station in stations_sub:
            sub_stations2.append(station)
    # 获得sub_stations2站点
    # 开始删除站点
    for station in stations:
        if station not in sub_stations2:
            stations_ans.append(station)
    # print('stations2\n', stations2[0:10])
    # 开始聚类
    label2stations = getStationClu(stations_ans, K=K)
    # print('label2stations\n', label2stations)
    labels = list(label2stations.keys())
    label_max = max(labels)
    # print('label_max', label_max)
    # print('labels:\n', list(label2stations.keys())[0:10])
    clu_cus = {}
    # print(label2stations)
    # print(label2stations[107])
    # 聚类簇 添加自定义簇
    for stations_sub in sub_stations:
        label_max = label_max+1
        stations_sub2 = []
        for stationId in stations_sub:
            stations_sub2.append([dic_staId_lnglat[str(stationId)][0], dic_staId_lnglat[str(stationId)][1], stationId])
        # print(label_max, stations_sub)
        label2stations[label_max] = stations_sub2
        clu_cus[label_max] = stations_sub2
    # print('/selectRegion 收到{0}个自定义聚类站点\n自定义聚类区域结果:{1}\n'.format(len(sub_stations), clu_cus))
    # 返回数据
    return label2stations, list(clu_cus.keys())

# 簇之间生成方式2
def getStationCluLink2(station2lables):
    # 簇之间连接的生成方式2, 试着用road_sta_end生成连接关系
    links = set()
    # 画图的关系
    road_sta_end = json_load(r'D:\pycharmProject\logic\dataSet\road_sta_end.json')
    road_sta_end = [[x[0][2], x[1][2]] for x in road_sta_end]
    # print('road_sta_end 长度:', len(road_sta_end))
    # 站点在哪个簇里？
    # station2lables = json_load(r'/4.站点聚类\station2lables.json')
    station2lables = station2lables
    # debug 传进来的station2lables和读取的station2lables是否一致---->一致
    # dic = {}
    # dic['station2lales'], dic['station2lables2'] = station2lables, station2lables2
    # json_dump(dic, 'station2lables.json')
    # debug 结束
    set_1 = set()
    for rate in road_sta_end:
        preSta, nextSta = str(rate[0]), str(rate[1])
        # 站点所在的簇存在吗
        if preSta not in list(station2lables.keys()): set_1.add(preSta)
        if nextSta not in list(station2lables.keys()): set_1.add(nextSta)
        if (preSta in list(station2lables.keys())) and (nextSta in list(station2lables.keys())):
            # 站点在两个不同的簇里
            if station2lables[preSta] != station2lables[nextSta]:
                links.add((station2lables[preSta], station2lables[nextSta]))
    return list(links)

def getNetData_antv(nodes, links):
    '''
    生成netData_antv数据
    :param nodes: 类{'id', 'x', 'y', 'degree'}
    :param edges: {'source', 'target'}
    :return:
    '''
    dic = {}
    dic['nodes'] = []
    dic['edges'] = []
    for node in nodes:
        dic['nodes'].append({
            'id': str(node['id']),
            "x": node['x'] if 'x' in list(node.keys()) else '',
            "y": node['y'] if 'y' in list(node.keys()) else '',
            "degree": node['degree'] if 'degree' in list(node.keys()) else ''
        })
    for link in links:
        dic['edges'].append({
            'source': str(link['staId']),
            'target': str(link['endId'])
        })
    return dic


if __name__ == '__main__':

    # cluAns = getStationClu(stations=json_load('D:\pycharmProject\logic\dataSet\staIdL.json'), K=110)
    # print(cluAns)
    # stations
    # 删去站点的聚类 站点+删除站点+聚类个数
    ans = getStationClu_sub(stations=json_load(r'D:\pycharmProject\logic\dataSet\staIdL.json'), sub_stations=[], K=110)
    pass












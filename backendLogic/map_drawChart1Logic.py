import random

import math

from temp import *
from comUsed.gpsXg import *

# 获取位置
def getPosition(label2stations, cluId2):
    '''
    生成位置
    :param: label2stations: cluId: stations
    :return: cluId: position
    '''

    dic = defaultdict(dict)
    for cluId, stations in label2stations.items():
        if cluId == cluId2:
            position = center_geolocation(stations)
            dic[cluId] = position

    return dic


# 获取站点客流 5~50
def getStationFlow(label2stations, timeSpan):
    '''
    站点客流
    :return: {上行: {stationId: flows, stationId: flows, stationId: flows}, 下行: {stationId: flows, stationId: flows, stationId: flows}}
    '''
    stationFlow = defaultdict(dict)
    # 模拟一下站点数据
    staL = json_load(r'D:\pycharmProject\busAnalyze-backend\data\staIdL.json')
    # 模拟上行
    for sta in staL:
        stationFlow[sta[2]]['up'] = 5+int(45*(random.random()))
        stationFlow[sta[2]]['down'] = 5+int(45*(random.random()))
    json_dump(stationFlow, r'D:\pycharmProject\busAnalyze-backend\data\stationFlow.json')
    return stationFlow


# 得到聚类站点中心点
def getLabel2Point(label2stations, labels):
    dic = {}
    # print(labels)
    json_dump(label2stations, 'tempjson.json')
    for label, stations in label2stations.items():
        if len(stations) != 0:
            dic[label] = center_geolocation(stations)
    return dic


# 得到聚类簇 外向和内向客流的站点
def getLabelDirectStations(label2stations, links, cluId):
    print('getLabelDirectStations label2stations', label2stations)
    print('getLabelDirectStations links', links)
    # 基本逻辑: 对每个link:{source: 1, target: 2}  簇1<->簇2 即stations1<->stations2
    # 每条线路的上行\下行站点序列
    road2stations = json_load(r'D:\pycharmProject\busAnalyze-backend\data\road2stations.json')
    # dic['12']['outStations']  dic['12']['inStations']
    dic_label_direct_stations = defaultdict(dict)
    # 找到连接关系
    links2 = []
    for link in links:
        # link1和link2
        link1 = [link['source'], link['target']]
        link2 = [link['target'], link['source']]
        links2.append(link1)
        links2.append(link2)
    for link in links2:
        label1, label2 = link[0], link[1]
        # if str(label1) != cluId and str(label2) != cluId:
        #     continue
        aList, bList = [], []
        # 每个连接关系 生成stations1和stations2
        stations1, stations2 = label2stations[int(link[0])], label2stations[int(link[1])]
        stations1, stations2 = [x[2] for x in stations1], [x[2] for x in stations2]
        stationSet1, stationSet2 = set(stations1), set(stations2)
        for road, stations in road2stations.items():
            direct = road.split('_')[1]
            # 此条路的站点序列:
            stations = [x[2] for x in stations]
            stationSet = set(stations)
            # 该路经过两个簇 确定有连接
            if len(stationSet&stationSet1)!=0 and len(stationSet&stationSet2)!=0:
                # 见ppt第9页 求取a和b
                # 遍历序列
                # print('线路序列', stations1, '\n', 'stations1', stations1, '\n', 'stations2', stations2)
                for i, station in enumerate(stations):
                    if i!=len(stations)-1:
                        if stations[i] in stations1 and stations[i+1] not in stations1:
                            a = station
                            aList.append([a, direct])
                        if stations[i] not in stations2 and stations[i+1] in stations2:
                            b = station
                            bList.append([b, direct])
        # 聚类外侧客流的站点 内测客流的站点
        dic_label_direct_stations[int(label1)]['outStation'] = aList
        dic_label_direct_stations[int(label2)]['inStation'] = bList
    print(dic_label_direct_stations)
    json_dump(dic_label_direct_stations, r'D:\pycharmProject\busAnalyze-backend\data\dic_lable_direct_stations.json')
    return dic_label_direct_stations

def getAngel(point, stationPos):
    # [经度, 纬度]
    x1, y1 = point[0], point[1]
    x2, y2 = stationPos[0], stationPos[1]
    # 弧度
    hudu = math.atan2( (y1-y2) , (x2-x1))
    # 角度
    angel = hudu*( 180 /math.pi)
    return angel

# 获取流量
def getFlow(label2stations, stationFlow, links, cluId):
    '''
    获取流量 输入: 聚类结果 站点客流 聚类图的连接结果
    :param: label2stations: cluId:{stations} stationFlow:站点客流 links:连接关系:{'source': 1, 'target': 2}
    :return: cluId: inflow[[angle, value]] outflow[[angle, value]]
    '''
    # 1.计算外围站点
    # dic_label_direct_stations = getLabelDirectStations(label2stations=json_load(r'D:\pycharmProject\busAnalyze-backend\tempData\label2stations2.json'), links=json_load(r'D:\pycharmProject\busAnalyze-backend\tempData\links.json'), cluId=12)
    dic_label_direct_stations = getLabelDirectStations(label2stations, links, cluId)

    # print("dic_label_direct_stations", dic_label_direct_stations)
    labels = list(dic_label_direct_stations.keys())
    # 计算角度 获得客流 [stationId]['out'][[angel, flow], [angel, flow]]
    # 2.获得label2Point
    label2Point = getLabel2Point(label2stations, labels)
    # print("label2Point", label2Point)
    dic_staId_lnglat = json_load(r'D:\pycharmProject\busAnalyze-backend\data\dic_staId_lnglat.json')
    staIdL = list(dic_staId_lnglat.keys())
    dic_label_direct_flow = defaultdict(dict)
    # 3.计算 [angel, flow]
    for label, dic_direct in dic_label_direct_stations.items():
        point = label2Point[label]
        for direct, stations in dic_direct.items():
            # 每个聚类簇(label)不同流向下:  中心位置point+station 生成一个 [angle, flow]
            # 求得 label_upOrIn => [[angel, flow]]
            dic_label_direct_flow[label]['in'] = []
            dic_label_direct_flow[label]['out'] = []
            for station in stations:
                if str(station[0]) not in staIdL:
                    continue
                angel = getAngel(point, dic_staId_lnglat[str(station[0])])
                angel = (360+angel) if angel < 0 else angel
                flow = stationFlow[station[0]][station[1]]
                if direct == 'outStation':
                    dic_label_direct_flow[label]['out'].append([flow, angel])
                else:
                    dic_label_direct_flow[label]['in'].append([flow, angel])
    return dic_label_direct_flow


# 获取指标
def getIndex(label2stations, timeSpan):
    '''
    获取聚类簇的指标
    :param label2stations: cluId: stations
    :param timeSpan: [开始时间, 结束时间]
    :return: cluId: [index1, index2]
    '''
    return 'index'



if __name__ == '__main__':

    ans = getLabelDirectStations(label2stations=json_load(r'D:\pycharmProject\busAnalyze-backend\tempData\label2stations2.json'), links=json_load(r'D:\pycharmProject\busAnalyze-backend\tempData\links.json'), cluId=12)
    print(ans)
    pass





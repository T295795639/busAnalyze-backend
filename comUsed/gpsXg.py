# 封装了一些有关gps的常用操作

import pandas
from comUsed.timeXg import timeSub
from geopy.distance import geodesic
from copy import deepcopy
from math import cos, sin, atan2, sqrt, pi, radians, degrees

def getGpsDis(pos1, pos2):
    '''
    两点距离:[经,纬]
    :param pos1:
    :param pos2:
    :return:
    '''
    return geodesic((pos1[1], pos1[0]), (pos2[1], pos2[0])).m

# 计算坐标簇的中心点
def center_geolocation(geolocations):
    x = 0
    y = 0
    z = 0
    lenth = len(geolocations)
    for loc in geolocations:
        lon, lat = loc[0], loc[1]
        lon = radians(float(lon))
        lat = radians(float(lat))

        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = float(x / lenth)
    y = float(y / lenth)
    z = float(z / lenth)

    return [degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y)))]


def getGpsAroundTime(gpsL, timeDif, pointIndex):
    '''
    取出点A周围时间的gps
    :param gpsL: 【点，时间】
    :param time: 时间半径
    :param pointIndex: 点A所在的索引
    :return: 取得的gps的索引 [index_t1, index_t2]
    '''
    point_time = gpsL[pointIndex][1]
    # 复制一份pointIndex 用于循环
    index = pointIndex
    index_t1 = index_t2 = -1
    while True:
        time = gpsL[index][1]
        if timeSub(point_time, time) > timeDif:
            index_t1 = index
            break
        index = index-1
    index = pointIndex
    while True:
        time = gpsL[index][1]
        if timeSub(time, point_time) > timeDif:
            index_t2 = index
            break
        index = index+1
    return index_t1, index_t2

def getGpsAroundPos(gpsL, radius, pointIndex):
    '''
    找半径内的gps点,好多点
    :param gpsL:
    :param radius:
    :param pointIndex:
    :return:
    '''
    if pointIndex >= len(gpsL):
        print('报错:点的索引超出范围')
    point = gpsL[pointIndex]
    # 复制一份pointIndex 用于循环
    index = pointIndex
    index_p1 = index_p2 = -1
    while True:
        point_pre = gpsL[index]
        if getGpsDis(point_pre, point) > radius:
            index_p1 = index
            break
        index = index-1
    index = pointIndex
    while True:
        try:
            point_next = gpsL[index]
        except:
            break
        if getGpsDis(point_next, point) > radius:
            index_p2 = index
            break
        index = index+1
    return index_p1, index_p2

def getLength(pathL):
    L = 0
    for i in range(len(pathL)-1):
        L = L + getGpsDis(pos1=pathL[i], pos2=pathL[i+1])
    return L

def getNei2Point(pointL, A):
    minIndex = -999
    minDis = 9e10
    minIndex2 = -999
    minDis2 = 9e10
    for i, point in enumerate(pointL):
        dis = getGpsDis(point, A)
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


def getRoadSpeed(roadGps):
    '''
    计算道路轨迹的车速
    :param roadGps: [点1, 点2, 点3, 点4]
    :return: speed矩阵 字典 '时间1-时间2': speed
    '''
    for i, gpsPoint in enumerate(roadGps):
        print(i)





if __name__ == '__main__':
    locations = [[116.568627,39.994879],[116.564791,39.990511]]
    coordinate = center_geolocation(locations)
    print(coordinate)
    pass
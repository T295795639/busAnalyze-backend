import matplotlib.pyplot as plt
import pandas as pd

from comUsed.gpsXg import *
from comUsed.paintXg import *
from comUsed.fileXg import *
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


## 测试与gps操作相关的方法



# 测试站点插入是否正确
def TestSta2Gps():
    dic_route2station = {}
    for path in getPath(r"D:\pycharmProject\logic\3.lineRoadNet\route2station")[1]:
        routeName = path.split("\\")[-1].replace(".json", '')
        List = json_load(path)
        List2 = []
        for x in List:
            if (x[2] > 100000):
                List2.append(x)
        dic_route2station[routeName] = List2
    df_gps = pd.read_csv(r"E:\output\GPS_oneTrip\2\706852\4.csv")
    gpsList = df_gps.loc[:, ['lng', 'lat']].values.tolist()
    stations = dic_route2station["2_down"]
    indexList, indexList_shift, gpsList = sta2Gps(gpsList=gpsList, staList=stations, type=0)
    drawSca(list=gpsList, color='blue')
    drawSca(list=stations, color='red', size=4)
    drawLine(list=gpsList, color='blue')
    plt.title('插入站点')
    plt.show()

# 测试 取得周围的点是否正确
def TestgetGpsAroundPos():
    # 1、取线路的站点
    dic_route2station = {}
    for path in getPath(r"D:\pycharmProject\logic\3.lineRoadNet\route2station")[1]:
        routeName = path.split("\\")[-1].replace(".json", '')
        List = json_load(path)
        List2 = []
        for x in List:
            if (x[2] > 100000):
                List2.append(x)
        dic_route2station[routeName] = List2
    stations = dic_route2station["2_down"]

    # 2、用站点分割线路
    df_gps = pd.read_csv(r"E:\output\GPS_oneTrip\2\706852\4.csv")
    gpsList = df_gps.loc[:, ['lng', 'lat']].values.tolist()
    # 站点插入到gps 返回插入前索引 插入后索引
    indexList, indexList_shift, gpsList = sta2Gps(gpsList=gpsList, staList=stations, type=0)
    # 画了一条路
    drawSca(list=gpsList, color='blue', size=1)
    # 画站点
    drawSca(list=stations, color='red')

    # 记录站点位置
    stations = []
    # 分割出每个站点的GPS 画出来
    for index_sta in indexList_shift:
        a, b = getGpsAroundPos(gpsL=gpsList, radius=10, pointIndex=index_sta)
        stations.append(a)
        stations.append(b)
        print(a, b)
        drawSca(list=gpsList[a:b], color='red', size=4)

    plt.title("画出站点附近10m内的点")
    plt.show()

    # 分割出每个路段
    sections = []
    for i, sta in enumerate(stations):
        if i == 0:
            sections.append([0, sta])
        elif i == len(stations) - 1:
            sections.append([sta, len(stations) - 1])
        elif i % 2 != 0:
            sections.append([sta, stations[i + 1]])
    for section in sections:
        drawSca(list=gpsList[section[0]:section[1]], color='blue')
    plt.title("画出路段")
    plt.show()


def TestgetRoadSpeed():
    pass

if __name__ == '__main__':
    TestSta2Gps()
    TestgetGpsAroundPos()
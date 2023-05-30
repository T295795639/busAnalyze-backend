from temp import *
from comUsed.gpsXg import *

# 测试路由函数功能
# 自定义聚类路由
def selectRegion():
    # 选择区域聚类 生成聚类结果
    # 请求到数据
    staL = json_load(r'D:\pycharmProject\busAnalyze-backend\testData\staL.json')
    stations = json_load(r'D:\pycharmProject\logic\dataSet\staIdL.json')
    # 删去一部分 开始聚类 聚类结果,label:stations
    label2stations, label_cus = getStationClu_sub(stations, staL, 110)
    print('selectRegion-label2stations-106', label2stations[106])
    station2label = {}
    # 遍历label,stations  lable:stations
    for label, stations2 in label2stations.items():
        for station in stations2:
            if type(station)==list:
                station2label[str(station[2])] = label
            else:
                station2label[station] = label
    # 生成连接关系 station2label
    links = getStationCluLink2(station2label)
    linksL = []
    for link in links:
        linksL.append({
            'staId': link[0],
            'endId': link[1]
        })

    # 通过label2stations 生成站点簇的中心点 107簇有问题 label:stations
    nodeL = []
    for label, stations in label2stations.items():
        # print(label, stations)
        point = center_geolocation(stations)
        nodeL.append({
            "id": label,
            "x": point[0],
            "y": point[1]
        })

    antvData = getNetData_antv(nodeL, linksL)
    antvData['label_cus'] = label_cus
    # 一个方法: 聚类(stations, sub_stations)
    # label2stations-->links
    return antvData

if __name__ == '__main__':
    # links
    # antvData = selectRegion()
    # print(antvData)

    staL = json_load(r'D:\pycharmProject\busAnalyze-backend\data\staIdL.json')
    dic = {}
    for sta in staL:
        dic[sta[2]] = [sta[0], sta[1]]
    json_dump(dic, 'dic_staId_loction.json')






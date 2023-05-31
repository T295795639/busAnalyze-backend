from flask import Flask, request, jsonify
from comUsed.gpsXg import *
from flask_cors import *
from temp import *
from backendLogic import map_drawChart1Logic

ret = {}
ret['start_time'] = ''
ret['end_time'] = ''

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

# 通过聚类结果 生成指标\外向客流\内向客流 chart1数据
@app.route('/map_drawChart1/<cluId>', methods=['GET', 'POST'])
def map_drawChart1(cluId):
    '''
    获得chart1数据
    聚类id 位置
    :return:
    cluId: {
        position: [],
        outFlow: [[angle, value], [angle1, value1]],
        inFlow: [[angle, value], [angle1, value1]],
        indexL: []
    }
    '''
    # 有一个簇 生成 流入该簇的客流和流出该簇的客流 入和出
    # 流入:簇外的站点-->簇内的站点  流出:簇内的站点-->簇外的站点
    global ret
    # print(ret['label2stations'])
    label2stations = ret['label2stations']
    # print(cluId)
    # print(label2stations)
    cluId = int(cluId)

    chart1_data = {}

    # 通过label2stations生成position
    dic_position = chart1_data['position'] = map_drawChart1Logic.getPosition(label2stations, cluId)

    # 站点客流 模拟
    stationFlow = map_drawChart1Logic.getStationFlow(label2stations, [ret['start_time'], ret['end_time']])

    # 通过label2stations生成客流信息
    dic_flow = chart1_data['flow'] = map_drawChart1Logic.getFlow(label2stations, stationFlow, ret['antvData']['edges'], cluId)

    # 通过label2stations生成指标信息
    dic_index = map_drawChart1Logic.getIndex(label2stations, [ret['start_time'], ret['end_time']])

    # 拼装数据
    # chart1_data = ''

    # print(dic_position)
    # print(chart1_data)
    json_dump(chart1_data['flow'], r'D:\pycharmProject\busAnalyze-backend\tempData\flow.json')
    return jsonify(chart1_data)


# 自定义聚类路由 存储网络数据
@app.route('/selectRegion', methods=['GET', 'POST'])
def selectRegion():
    global ret
    # 选择区域聚类 生成聚类结果
    if request.method == 'POST':
        res_data = request.json
        json_dump(res_data, r'D:\pycharmProject\busAnalyze-backend\testData\选择区域聚类\1.Q_stationSub_cluNum.json')
        staL, cluNum = res_data[0:-1], res_data[-1]
        cluNum = cluNum - len(staL)
        stations = json_load(r'D:\pycharmProject\logic\dataSet\staIdL.json')
        # 删去一部分 开始聚类 聚类结果,label:stations
        label2stations, label_cus = getStationClu_sub(stations, staL, cluNum)
        json_dump(label2stations, r'D:\pycharmProject\busAnalyze-backend\testData\选择区域聚类\cluAns.json')
        station2label = {}
        # 遍历label,stations  lable:stations
        for label, stations2 in label2stations.items():
            for station in stations2:
                if type(station) == list:
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
        ret['label2stations'] = label2stations
        ret['antvData'] = antvData
        # 一个方法: 聚类(stations, sub_stations)
        # label2stations-->links
        json_dump(antvData, r'D:\pycharmProject\busAnalyze-backend\testData\选择区域聚类\2.H_antvData.json')
        # 触发chart1数据的改变
        return jsonify(antvData)

# 获取站点位置
@app.route('/getStationPos', methods=['GET', 'POST'])
def getStationPos():
    if request.method == 'POST':
        staId = request.json['stationId']
        dic_staId_loc = json_load(r'D:\pycharmProject\busAnalyze-backend\data\dic_staId_lnglat.json')
        pos = dic_staId_loc[staId]
        return pos
    else:
        return '1'

# 获取线路绘制数据
@app.route('/getRouteData/<routeName>')
def getRouteData(routeName):
    '''
    获取线路数据
    :param routeName: 线路名
    :return: 绘制数据
    '''
    # 原始数据 {'102_down': [point1, point2, point3]}
    dic_route = {}



    route_geoJosn = {"返回": 'ok'}
    return jsonify(route_geoJosn)


if __name__ == '__main__':
    app.run()







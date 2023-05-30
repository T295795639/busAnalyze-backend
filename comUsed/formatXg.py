##  **************************************************格式转换*********************************************
from temp import *
# geojson相关

def getGeojson(data, type):
    # type 可选 LineString,Point
    features = []
    for item in data:
        features.append({
            'type': 'Feature',
            'properties': {},
            'geometry': {
                'type': type,
                'coordinates': item
            }
        })
    geoJson = {
      "type": "FeatureCollection",
      "features": features
    }
    return geoJson

def getNetData_antv(nodes, links):
    '''
    生成netData_antv数据
    :param nodes: 类{'id', 'x', 'y', 'degree'}
    :param links: {'source', 'target'}
    :return:
    '''
    dic = {}
    dic['nodes'] = []
    dic['edges'] = []
    for node in nodes:
        dic['nodes'].append({
            'id': str(node.id),
            "x": node.x if hasattr(node, 'x') else '',
            "y": node.y if hasattr(node, 'y') else '',
            "degree": node.degree if hasattr(node, 'degree') else ''
        })
    for link in links:
        dic['edges'].append({
            'source': str(link.staId),
            'target': str(link.endId)
        })
    return dic

def getNetData_echarts(nodes, links):
    dic = {}
    dic['nodes'] = []
    dic['links'] = []
    for node in nodes:
        dic['nodes'].append({
            'id': node['id'],
            'name': 'name',
            "symbolSize": 8,
            'x': node['x'],
            'y': node['y']
        })
    for link in links:
        dic['links'].append({
            'source': link[0],
            'target': link[1]
        })
    return dic

# dic格式相关
def dicKeySort(dic):
    ## 字典key排序
    dic2 = {}
    cluIdL = [int(x) for x in list(dic.keys())]
    cluIdL = sorted(cluIdL)
    for cluId in cluIdL:
        cluId = str(cluId)
        dic2[cluId] = dic[cluId]
    return dic2

def getEdges(rateData):
    edges = rateData["edges"]
    Edges = []
    for edge in edges:
        Edges.append([edge["source"], edge["target"]])
    return Edges

if __name__ == '__main__':

    data1 = json_load(r'D:\pycharmProject\logic\dataSet\road_sta_end.json')
    data2 = json_load(r'D:\pycharmProject\logic\dataSet\staIdL.json')
    geoJson_line = getGeojson(data1, 'LineString')
    geoJson_point = getGeojson(data2, 'Point')
    json_dump(geoJson_line, 'line.json')
    json_dump(geoJson_point, 'point.json')




import os
import json

# 得到文件名字和路径
def getPath(f):
    fNameL = os.listdir(f)
    fPathL = [os.path.join(f, x) for x in fNameL]
    return fNameL, fPathL

def json_load(filename):
    return json.load(open(filename, encoding='utf-8'))


def json_dump(data, tof):
    return json.dump(data, open(tof, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

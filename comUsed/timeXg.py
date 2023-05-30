from datetime import datetime

def timeSub(time1, time2):
    a = datetime.strptime(str(time1), "%Y/%m/%d %H:%M:%S.%f")
    b = datetime.strptime(str(time2), "%Y/%m/%d %H:%M:%S.%f")
    return (a-b).seconds

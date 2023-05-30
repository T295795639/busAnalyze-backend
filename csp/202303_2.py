# 垦田计划
# 70分 错误:超时

def getMax(areas):
    hours = [x[0] for x in areas]
    return (hours.index(max(hours)))


if __name__ == '__main__':
    # 输入: n, m, k
    # 输出: 最少耗时

    # 待开垦区域总数, 资源总量, 最少开垦天数
    n, m, k = map(int, input().split())
    areas = [[i for i in map(int, input().split())] for j in range(n)]

    # 不断给最大的减一
    while True:
        maxIndex = getMax(areas)
        hour, cost = areas[maxIndex]
        if m>=cost:
            m-=cost
            hour-=1
            areas[maxIndex][0] = hour
        # 退出条件:最大时间低于阈值
        if hour <= k:
            break
        # 退出条件:不能再减了
        if m < cost:
            break
    print(areas[getMax(areas)][0])

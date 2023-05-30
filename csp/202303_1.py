# 建议使用穷举法
# 输入 n, a, b 田地个数 右上角坐标(n, n)
# n行
# 20分

def isInAB(x, y, a, b):
   if (x<a and y<b and x>0 and y>0):
       return True
   else:
       return False

# print(isInAB(5, 5, 10, 10))

n, a, b = map(int, input().split())
points = [[i for i in map(int, input().split())] for j in range(n)]

# 如果一个点在(a, b)内 计算截取面积
# 如果两个点在(a, b)内 计算面积
# 如果两个点都不在(a, b)内 面积为零
sum = 0
for point in points:
    x1, y1, x2, y2 = point[0], point[1], point[2], point[3]
    s = 0
    # print(x1, y1, x2, y2)
    # 两个点都在内
    if isInAB(x1, y1, a, b) and isInAB(x2, y2, a, b):
        # print('都在内', (x2-x1)*(y2-y1))
        s = (x2-x1)*(y2-y1)
    # 都不在内
    elif not isInAB(x1, y1, a, b) and not isInAB(x2, y2, a, b):
        # 不在内 但是面积不为零的情况
        if x1<a and y2>0 and y1<0 and x2>b:
           s = (a-max(x1, 0))*min(y2, b)
        # elif x1<0 :
        else:
            s = 0
    # 只有一个在
    elif isInAB(x1, y1, a, b) or isInAB(x2, y2, a, b):
        # print('只有一个在')
        # 左边的点在里面
        if isInAB(x1, y1, a, b):
            # print((a-x1)*(b-y1))
            s = (min(a, x2)-x1)*(min(b, y2)-y1)
        if isInAB(x2, y2, a, b):
            # print(x2*y2)
            s = (x2-max(x1, 0))*(y2-max(y1, 0))
    # print(point, s)
    sum += s
print(sum)

# 样例输入
# 4 10 10
# 0 0 5 5
# 5 -2 15 3
# 8 8 15 15
# -2 10 3 15





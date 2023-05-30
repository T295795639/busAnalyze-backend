# 训练计划

def getPre():
    pass


if __name__ == '__main__':
    # 距离开赛的天数和训练科目m,n

    # 最早开始时间：该科目最早可以于哪一天开始训练？ 根据依赖的科目决定
    # 最晚开始时间：在不耽误参赛的前提下（n天内完成所有训练），该科目最晚可以从哪一天开始训练？ 根据耗时和开赛的天数决定

    m, n = map(int, input().split())
    # 依赖列表
    yilaiList = [i for i in map(int, input().split())]
    # 耗时列表
    hourList = [i for i in map(int, input().split())]

    print(m, n)
    print(yilaiList)
    print(hourList)

    # 计算最早开始时间
    # 对每一个科目,有依赖否？有:加依赖 无:开始时间为1  开始!

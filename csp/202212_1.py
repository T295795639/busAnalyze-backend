# 设计程序 计算盈利和亏损

n, i = map(float, input().split())
n = int(n)

# 收入+支出
moneys = [int(x) for x in input().split()]
money_pre = moneys[0]

year = 0
sumMoney = 0

for j, money in enumerate(moneys):
    # 盈利 大于0且不是第一年
    if sumMoney > 0 and j!=0:
        sumMoney *= (1+i)
    sumMoney += money*((1+i)**(-year))
    year += 1


print("{0:.3f}".format(sumMoney))





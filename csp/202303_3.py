# 第三题

# 1:2 代表第一个属性1为2的DN与之匹配
# 3~1 代表拥有属性3且属性3不为1的DN
# 逻辑组合情况: |(表达式1)(表达式2)  &(表达式1)(表达式2)
# 没分

def getAttrValue(user, attrIndex):
    if user[1]==0:
        return False
    for i in range(2, len(user)):
        if i%2 == 0:
            if user[i] == attrIndex:
                return user[i+1]
    return False

# 1:2
def getCase1(users, str):
    ans = []
    # 解构
    attrIndex, value = map(int, str.split(':'))
    for user in users:
        if getAttrValue(user, attrIndex) == value:
            ans.append(user[0])
    return ans

# 1~2
def getCase2(users, str):
    ans = []
    # 解构
    attrIndex, value = map(int, str.split('~'))
    for user in users:
        if getAttrValue(user, attrIndex)!=False and getAttrValue(user, attrIndex)!=value:
            ans.append([user[0]])
    return ans

# &(1:2)(1~2)
def getCase3(users, str):
    cases = [item.replace('(', '') for item in str[1:].split(')')[:-1]]
    ans = []
    for case in cases:
        if case.find(':')!=-1:
            ans.append(getCase1(users, case))
        if case.find('~')!=-1:
            ans.append(getCase2(users, case))
    return list(set(ans[0]) & set(ans[1]))

# |(1:2)(1~2)
def getCase4(users, str):
    cases = [item.replace('(', '') for item in str[1:].split(')')[:-1]]
    ans = []
    for case in cases:
        if case.find(':')!=-1:
            ans.append(getCase1(users, case))
        if case.find('~')!=-1:
            ans.append(getCase2(users, case))
    return list(set(ans[0]) | set(ans[1]))


if __name__ == '__main__':

    n = int(input())
    users = [[j for j in map(int, input().split())] for i in range(n)]
    m = int(input())
    strs = []
    for i in range(m):
        strs.append(input())

    ans = []
    for str in strs:
        # 4种情况
        # 1:2
        if str.find(':')!=-1 and str.find('&')==-1 and str.find('|')==-1:
            ans.append(getCase1(users, str))
        # 1~2
        if str.find('~')!=-1 and str.find('&')==-1 and str.find('|')==-1:
            ans.append(getCase2(users, str))
        # &(1:2)(2:3)
        if str.find('&')!=-1:
            ans.append(getCase3(users, str))
        # |(1:2)(2:3)
        if str.find('|')!=-1:
            ans.append(getCase4(users, str))

    for one_ans in ans:
        one_ans.sort()
        if len(one_ans)==0:
            print()
        else:
            for i in one_ans:
                if i == len(one_ans):
                    print(i, end='')
                else:
                    print(i, end=' ')
            print()





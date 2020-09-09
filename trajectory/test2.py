def LongestCommonSubsequence(a, b):
    lena = len(a)
    lenb = len(b)
    # c用来保存对应位置匹配的结果
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    # flag用来记录转移方向
    flag = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:  # 字符匹配成功，则该位置的值为左上方的值加1
                c[i + 1][j + 1] = c[i][j] + 1
                flag[i + 1][j + 1] = 'ok'
            elif c[i + 1][j] > c[i][j + 1]:   # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                c[i + 1][j + 1] = c[i + 1][j]
                flag[i + 1][j + 1] = 'left'
            else:   # 上值大于左值，则该位置的值为上值，并标记方向up
                c[i + 1][j + 1] = c[i][j + 1]
                flag[i + 1][j + 1] = 'up'

    (p1, p2) = (len(a), len(b))
    s = []
    while c[p1][p2]:    # 不为None时
        t = flag[p1][p2]
        if t == 'ok':   # 匹配成功，插入该字符，并向左上角找下一个
            s.append(a[p1-1])
            p1 -= 1
            p2 -= 1
        if t == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if t == 'up':   # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return s

if __name__ == '__main__':
    a = [180, 180, 141.1, 146, 141, 200, 235, 235, 173, 172, 141, 141, 172, 180]
    b = [165, 235, 180.2, 141, 240, 171, 173.5, 172.2, 141.5]

    s = LongestCommonSubsequence(a, b)
    print(s)
    print(len(s)/min(len(a), len(b)))
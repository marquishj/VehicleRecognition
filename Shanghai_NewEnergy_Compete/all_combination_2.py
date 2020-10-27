# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:44
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_1.py
# @Software: PyCharm
'''python 数组里面求和为某固定值的所有组合？'''
'''https://zhidao.baidu.com/question/205625312281053725.html'''

# l = [l for l in range(8)]
# l=[1,2,3,4,5,6,7,8]
l = [2,3,4,5,6,7,8,10,12,13,23,34,56]
# print(l)

def combination(l, n):
    l = list(sorted(filter(lambda x: x <= n, l)))
    combination_impl(l, n, [])


def combination_impl(l, n, stack):
    if n == 0:
        print(stack)
        return
    for i in range(0, len(l)):
        if l[i] <= n:
            stack.append(l[i])
            combination_impl(l[i + 1:], n - l[i], stack)
            stack.pop()
        else:
            break


combination(l, 10)
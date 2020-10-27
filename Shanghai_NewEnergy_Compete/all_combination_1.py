# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:44
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_1.py
# @Software: PyCharm

l = [2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 23, 34, 56]


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


combination(l, 50)
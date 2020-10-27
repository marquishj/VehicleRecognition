# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 18:12
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : blank.py
# @Software: PyCharm

label=[1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
label_0=list([0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0])
label_1=list([0,1,1,0,0,2,1,2,2,0,2,1,0,0,0,2,2,1,1,0,0,1,0,1,1])

m=0
n=0
for i in label:
    if i == label_0[n]:
        m = m + 1
    n = n + 1
acc_0 = m / 25
print('acc_0: {}'.format(acc_0))

m=0
n=0
for i in label:
    # for j in range(len(label_1)):
    if i == label_1[n] :
        m = m + 1
    n=n+1
acc_1=m/25
print('acc_1: {}'.format(acc_1))



# mylist = list1.split(',')
# print( mylist)

# price = [x.strip() for x in list1 if x.strip() != '']
# print( price)
# for i in list1:
#    if i == ', ':
#         list1.remove(' ')
# print( list1)
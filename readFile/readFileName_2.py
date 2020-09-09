# coding=utf-8
import os
import xlwt  # 操作excel模块
import sys

sys.path[0]='F:\\sql data\\Step5_data_process_5_sql_20200708'
# file_path = sys.path[0] + '\\filenamelist.xls'  # sys.path[0]为要获取当前路径，filenamelist为要写入的文件
file_path = sys.path[0] + '\\filenamelist.csv'  # sys.path[0]为要获取当前路径，filenamelist为要写入的文件
f = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个excel
sheet = f.add_sheet('sheet1')  # 新建一个sheet
pathDir = os.listdir(sys.path[0])  # 文件放置在当前文件夹中，用来获取当前文件夹内所有文件目录

i = 0  # 将文件列表写入test.xls
for s in pathDir:
    sheet.write(i, 0, s)  # 参数i,0,s分别代表行，列，写入值
    i = i + 1

print(file_path)
print(i)  # 显示文件名数量
f.save(file_path)
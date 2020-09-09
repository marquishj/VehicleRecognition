'''https://download.csdn.net/download/weixin_43375239/11156457'''
import xlrd # 读xlsx
import xlsxwriter # 写xlsx
import urllib.request # url请求，Python3自带
import os # 创建output文件夹
import glob # 获取文件夹下文件名称
import time # 记录时间
import json # 读取json格式文件

# 本函数完成文件合并。
def xlsx_merge(folder,header,filename):
    fileList = []
    for fileName in glob.glob(folder + "*.xlsx"):
        fileList.append(fileName)
    fileNum = len(fileList)
    matrix = [None] * fileNum
    for i in range(fileNum):
        fileName = fileList[i]
        workBook = xlrd.open_workbook(fileName)
        try:
            sheet = workBook.sheet_by_index(0)
        except Exception as e:
            print(e)
        nRows = sheet.nrows
        matrix[i] = [0]*(nRows - 1)
        nCols = sheet.ncols
        for m in range(nRows - 1):
            matrix[i][m] = ["0"]* nCols
        for j in range(1,nRows):
            for k in range(nCols):
                matrix[i][j-1][k] = sheet.cell(j,k).value
    fileName = xlsxwriter.Workbook(folder + filename + ".xlsx")
    sheet = fileName.add_worksheet("merged")
    for i in range(len(header)):
        sheet.write(0,i,header[i])
    rowIndex = 1
    for fileIndex in range(fileNum):
        for j in range(len(matrix[fileIndex])):
            for colIndex in range (len(matrix[fileIndex][j])):
                sheet.write(rowIndex,colIndex,matrix[fileIndex][j][colIndex])
            rowIndex += 1
    print("已完成%d个文件的合并"%fileNum)
    fileName.close()

# 本函数完成获取POI
def poi_by_adcode_poicode(folder,city_file = "polygon",poi_file = "poi",result_file = "result",merge_or_not = 1):
    
    key="ffaca7d2831889ce6857092abb76ce5a"
    count=0
    city_file = city_file
    poi_file = poi_file
    merge_or_not = merge_or_not
    header_full = ["id","name","type","typecode","biz_type","address","location","tel","pname","cityname","adname","rating","cost"]
    header = ["id","name","type","typecode","biz_type","address","location","tel","pname","cityname","adname"]
    offset = 10 # 实例设置每页展示25条
    output_folder = folder + "output\\"
    # 创建输出路径
    if os.path.isdir(output_folder):
        pass
    else:
        os.makedirs(output_folder)
    # 读取列表
    city_sheet =  xlrd.open_workbook(folder+ "input\\" + city_file + ".xlsx").sheet_by_index(0)
    poi_type_sheet = xlrd.open_workbook(folder+ "input\\" + poi_file + ".xlsx").sheet_by_index(0)

    city_list =city_sheet.col_values(0)
    city_code_list = city_sheet.col_values(1)
    upleftjd=city_sheet.col_values(1)
    upleftwd=city_sheet.col_values(2)
    rightbottomjd=city_sheet.col_values(3)
    rightbottomwd=city_sheet.col_values(4)
    jd = city_sheet.col_values(6)
    wd = city_sheet.col_values(7)
    poi_type_list = poi_type_sheet.col_values(1)
    poi_type_name=poi_type_sheet.col_values(0)
    result_file = result_file+str(poi_type_name[1])
    # 指示工作完成量
    total_work = (city_sheet.nrows - 1)  * (poi_type_sheet.nrows - 1)
    work_index = 1
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + "：抓取开始！")
    for city_index in range(1,len(city_list)):
        for poi_type_index in range(1,len(poi_type_list)):
            workbook =xlsxwriter.Workbook(output_folder + str(city_list[city_index]) +"_"+
                                          str(poi_type_list[poi_type_index])+"_"+
                                          str(jd[city_index])+"_"+str(wd[city_index])+ ".xlsx") # 新建工作簿
            sheet = workbook.add_worksheet("result") # 新建“poiResult”的工作表
            for col_index in range(len(header_full)):
                sheet.write(0,col_index,header_full[col_index]) # 写表头
            row_index = 1
            for page_index in range(1, 101):
                try:
                    
                    url = "http://restapi.amap.com/v3/place/polygon?types=" + \
                          str(poi_type_list[poi_type_index]) + "&polygon=" + \
                          str(round(upleftjd[city_index],6))+","+ \
                          str(round(upleftwd[city_index],6))+"|"+ \
                          str(round(rightbottomjd[city_index],6))+","+\
                          str(round(rightbottomwd[city_index],6))+"&offset=" + \
                          str(offset) + "&page="+ str(page_index) +"&key="+str(key)+\
                          "&extensions=all&output=json"
                    data = json.load(urllib.request.urlopen(url))["pois"]
                    count=count+1
                    for i in range(offset):
                        for col_index in range(len(header)):
                            sheet.write(row_index, col_index, str(data[i][header[col_index]]))
                            sheet.write(row_index,len(header),str(data[i]["biz_ext"]["rating"]))
                            sheet.write(row_index,len(header) + 1,str(data[i]["biz_ext"]["cost"]))
                        row_index += 1 
                except:
                    break
            print("已完成：" + str(poi_type_list[poi_type_index]))
            workbook.close()   
            print(str(city_list[city_index]) + " " + str(poi_type_list[poi_type_index] )+
                  " 已获取!进度：%.2f%%"  %(work_index / total_work *100))
            work_index += 1
    print( "所有地区各类别POI获取完毕")
    print("搜索次数："+str(count))
    if merge_or_not == 1:
        xlsx_merge(output_folder, header_full, result_file)
        print("已对文件进行合并！")
    else:
        print("未进行合并！")
    print("所有工作完成！")

# 使用 参数分别为：路径、格网划分excel、poi类型excel、输出文件
# poi_by_adcode_poicode("E:/poi/","搜索区域格网划分", "poi类型", "输出", 1)
poi_by_adcode_poicode("F:\\PycharmProjects\\VehicleRecognition\\POI\\GetAmapPOIbyPolygon\\",
                      "搜索区域格网划分-Shanghai", "poi类型-Shanghai", "输出", 1)

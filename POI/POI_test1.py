import xlwt #创建Excel，见代码行8，9，11，25，28；CMD下：运行pip install xlwt进行安装
import urllib.request # url请求，Python3自带，Python2与3中urllib的区别见：http://blog.csdn.net/Jurbo/article/details/52313636
from bs4 import BeautifulSoup # 快速获取网页标签内容的库；CMD下：运行pip install beautifulsoup4进行安装
import re # 使用正则表达式的库，代码行7，快速学习见：http://www.runoob.com/regexp/regexp-syntax.html


poiTag = ["id","name","type","typecode","biz_type","address","location","tel","pname","cityname","adname"] #返回结果控制为base时，输出的POI标签类别
poiSoupTag = ["idSoup","nameSoup","typeSoup","typecodeSoup","biz_typeSoup","addressSoup","locationSoup","telSoup","pnameSoup","citynameSoup","adnameSoup"] #包装对应的Soup
pattern = re.compile("(?:>)(.*?)(?=<)",re.S) # 组织正则表达式
poiExcel =xlwt.Workbook() # 新建工作簿
sheet = poiExcel.add_sheet("poiResult") # 新建“poiResult”的工作表

for colIndex in range(len(poiTag)):
    sheet.write(0,colIndex,poiTag[colIndex]) # 写表头

offset = 10 # 实例设置每页展示10条POI（官方限定25条）
maxPage = 10 # 设置最多页数为10页（官方限定100页）
types = "090000" # 示例类别为医疗保健服务POI，下载：http://a.amap.com/lbs/static/zip/AMap_poicode.zip
city = "440305" # 示例类别为深圳市南山区，下载：http://a.amap.com/lbs/static/zip/AMap_adcode_citycode.zip

for pageIndex in range(1, maxPage + 1):
    try:
        # url = "http://restapi.amap.com/v3/place/text?&keywords=&types=" + types + "&city=" + city + "&citylimit=true&output=xml&offset=" + str(offset) + "&page="+ str(pageIndex) + "&key=GANo2ajua2rTiTTc05oKIrYr&extensions=base"
        # url = "https://restapi.amap.com/v3/place/around?key=GANo2ajua2rTiTTc05oKIrYr&location=116.473168,39.993015&radius=10000&types=011100" + types + "&city=" + city + "&citylimit=true&output=xml&offset=" + str(offset) + "&page="+ str(pageIndex) + "&key=GANo2ajua2rTiTTc05oKIrYr&extensions=base"
        url = "http://restapi.amap.com/v3/place/around?key=GANo2ajua2rTiTTc05oKIrYr&location=116.473168,39.993015&keywords=&types=011100&radius=1000&offset=20&page=1&extensions=all"
        # 请求的结构化url地址如上；请使用自己的key，见：http://lbs.amap.com/api/webservice/guide/api/search/
        poiSoup = BeautifulSoup(urllib.request.urlopen(url).read(),"xml") #读入对应页码的页面

        for tagIndex in range(len(poiTag)):
            poiSoupTag[tagIndex] = poiSoup.findAll(poiTag[tagIndex]) # 根据Tag读对应页码的POI标签内容

        for rowIndex in range(len(poiSoupTag[0])):
            for colIndex in range(len(poiSoupTag)):
                sheet.write(len(poiSoupTag[0]) * (pageIndex - 1) + rowIndex + 1, colIndex, re.findall(pattern,str(poiSoupTag[colIndex][rowIndex])))
                # 根据正则表达式提取内容，并在对应行与列写入
    except Exception as e:
        print(e) # 设置错误输出

poiExcel.save("F:\\sql data\\POI\\&" + types + "&" + city + ".xls") # 保存
print("Done!") # 结束
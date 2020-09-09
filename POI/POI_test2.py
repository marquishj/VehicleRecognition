import xlwt #创建Excel，见代码行8，9，11，25，28；CMD下：运行pip install xlwt进行安装
import urllib.request # url请求，Python3自带，Python2与3中urllib的区别见：http://blog.csdn.net/Jurbo/article/details/52313636
from bs4 import BeautifulSoup # 快速获取网页标签内容的库；CMD下：运行pip install beautifulsoup4进行安装
import re # 使用正则表达式的库，代码行7，快速学习见：http://www.runoob.com/regexp/regexp-syntax.html

url = "http://restapi.amap.com/v3/place/text?&keywords=&types=" + self.cityType + "&city=" + self.cityID + "&citylimit=true&output=xml&offset=" + str(
    offset) + "&page=" + str(maxPage) + "&key=你的key&extensions=base"

if self.getHtml(url) == 0:
    print
    'parsing page 1 ... ...'
    # parse the xml file and get the total record number
    totalRecord_str = self.parseXML()

    totalRecord = string.atoi(str(totalRecord_str))
    if (total_record % offset) != 0:
        maxPage = totalRecord / offset + 2
    else:
        maxPage = totalRecord / offset + 1

    print(totalRecord)
    print(maxPage)

for pageIndex in range(1, maxPage + 1):
    try:
        url = "http://restapi.amap.com/v3/place/text?&keywords=&types=" + self.cityType + "&city=" + self.cityID + "&citylimit=true&output=xml&offset=" + str(
            offset) + "&page=" + str(pageIndex) + "&key=你的key&extensions=base"
        # 请求的结构化url地址如上；请使用自己的key，见：http://lbs.amap.com/api/webservice/guide/api/search/
        poiSoup = BeautifulSoup(urllib.urlopen(url).read(), "xml")  # 读入对应页码的页面
        for tagIndex in range(len(poiTag)):
            poiSoupTag[tagIndex] = poiSoup.findAll(poiTag[tagIndex])  # 根据Tag读对应页码的POI标签内容

        for rowIndex in range(len(poiSoupTag[0])):
            for colIndex in range(len(poiSoupTag)):
                sheet.write(len(poiSoupTag[0]) * (pageIndex - 1) + rowIndex + 1, colIndex,
                            re.findall(pattern, u" " + str(poiSoupTag[colIndex][rowIndex])))
                # 根据正则表达式提取内容，并在对应行与列写入
    except Exception as e:
        print(e)  # 设置错误输出

poiExcel.save("E:/POI&" + self.cityType + "&" + self.cityID + ".xls")  # 保存

poiExcel.save("F:\\sql data\\POI\\&" + types + "&" + city + ".xls") # 保存
print("Done!") # 结束
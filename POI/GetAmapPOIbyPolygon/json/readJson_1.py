import json
import pandas as pd
import numpy as np


f = open('F:\\PycharmProjects\\BRI_202008\\src\\SUE\\MATLAB\\createExcel\\POI\GetAmapPOIbyPolygon\\json\\shanghai_area_bak_amap.json',encoding='utf-8')
content = f.read() #使用loads（）方法需要先读文件
data_json = json.loads(content)
print(data_json)

f1 = open('F:\\PycharmProjects\\BRI_202008\\src\\SUE\\MATLAB\\createExcel\\POI\GetAmapPOIbyPolygon\\json\\shanghai_area_bak_amap.json',encoding="utf-8")
user_dic = json.load(f1)
print(user_dic)
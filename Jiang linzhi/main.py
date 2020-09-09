import numpy as np
import pandas as pd
import sys
sys.path.append(r"F:\\PycharmProjects\\VehicleRecognition\\Jiang linzhi")
import function



np_1D_array=np.array([1,2,3])
np_2D_array=np.array([[1,2,3],[4,5,6]])
np_3D_array = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])



result=function.price_fixed_Optimization(np_1D_array,np_2D_array,np_3D_array)
print(result)


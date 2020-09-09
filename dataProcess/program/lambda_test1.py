import numpy as np
import pandas as pd

y=map(lambda x:x+1,[1,2,3])
# z=pd.reduce(lambda x:x+1,[1,2,3])
print(list(y))
from functools import reduce
lst = [1, 2, 3, 4, 5]

f_res = filter(lambda x: x>3, lst)
r_res = reduce(lambda x, y: x*y, lst)

print(f_res)
print(r_res)


res = list(map(lambda x:x*x, lst))
print(res)
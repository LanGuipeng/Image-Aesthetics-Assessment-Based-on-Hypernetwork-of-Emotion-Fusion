#######分数映射
import numpy as np
import math

from sympy import C


def map1(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = np.max(data)    # 当前数据最大值
    d_max = np.min(data)    # 当前数据最小值
    return MIN +(MAX-MIN)/(d_max-d_min) * (data - d_min)

# data=[0.1,0.5,0.9,0.3]
# s=map(data,1,10)
# o=[]
# for i in s:
#     i=int(i)
#     o.append(i)
# print(s)
# print(o)


import pandas as pd
from collections import Counter
high=[]
low=[]
path = '/myDockerShare/zys/aes-emotion/experiment/excitement.csv'
# 使用pandas读入
data = pd.read_csv(path) #读取文件中所有数据
# 按列分离数据
x = data['score'].tolist()
A=list()            #不是元组数据，仍未lisT
for b in x:
    if b>0:
        c=b
        A.append(c)
    else:
        c=-b
        A.append(c)
# print(A)

s=map1(A,1,10)
score = list(map(int, s[:]))
# print(score)
count = Counter(score)
# 输出元素3的个数
print('分数为1的个数:',count[1]) 
print('分数为2的个数:',count[2]) 
print('分数为3的个数:',count[3]) 
print('分数为4的个数:',count[4]) 
print('分数为5的个数:',count[5]) 
print('分数为6的个数:',count[6]) 
print('分数为7的个数:',count[7]) 
print('分数为8的个数:',count[8]) 
print('分数为9的个数:',count[9]) 
print('分数为10的个数:',count[10]) 

# print(s)
for i in s:
    if i<5:
        low.append(i)
    else:
        high.append(i)
print('高质量数量：',len(high))
print('低质量数量：',len(low))



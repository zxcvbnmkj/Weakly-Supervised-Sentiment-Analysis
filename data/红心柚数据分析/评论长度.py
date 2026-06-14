import numpy as np
import pandas as pd

df=pd.read_csv("./评论长度.csv")
# def connect(str):
#     str=str.replace(" ",'')
#     str=str.replace('\n', '')
#     return len(str)
#
#
# df2 = pd.DataFrame(columns=['评论长度'])
# df2["评论长度"] = df.分词.apply(connect)
# df2.to_csv("./评论长度.csv",index=False)

# print(max(df["评论长度"]))
hist, bins = np.histogram(df["评论长度"], bins=1, range=(1, 10))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(11, 20))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(21, 30))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(31, 40))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(41, 50))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(51, 100))
print(hist)
print(bins)

hist, bins = np.histogram(df["评论长度"], bins=1, range=(101, 357))
print(hist)
print(bins)

"""
[2240]
[ 1. 10.]
[2353]
[11. 20.]
[997]
[21. 30.]
[832]
[31. 40.]
[1374]
[41. 50.]
[1458]
[ 51. 100.]
[232]
[101. 357.]
"""



import pandas as pd
from matplotlib import pyplot as plt

# df=pd.read_csv("./分词后.csv")
# df=df["评论时间"]
# df.to_csv("./评论时间.csv",index=False)

intervals = ['2017年及以前', '2018年', '2019年', '2020年', '2021年',"2022年","2023年"]
data = [111, 260, 284, 1019, 1386,1113,5313]

plt.bar(intervals, data)
plt.xticks(FontProperties="STSong")
# 设置标题和坐标轴标签
# plt.title('',FontProperties="STSong")
plt.xlabel('时间区间',FontProperties="STSong")
plt.ylabel('评论数',FontProperties="STSong")
plt.savefig("./时间区间图.jpg")
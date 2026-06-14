import pandas as pd
from matplotlib import pyplot as plt
import seaborn

#绘制分数分布图
df=pd.read_csv("./分词后.csv")
# graph=seaborn.countplot(x='评分', data=df)
# plt.xlabel('评分', fontsize=20, fontweight='bold',fontproperties="STSong")
# plt.ylabel('数量', fontsize=20, fontweight='bold',fontproperties="STSong")
# #加这一行可以在柱形图上显示出具体的数字
# graph.bar_label(graph.containers[0])
# plt.savefig("分数分布图.jpg")

intervals = ['0-10', '10-20', '20-30', '30-40', '40-50']
data = [15, 10, 8, 12, 5]
import matplotlib.pyplot as plt

# 创建柱状图
plt.bar(intervals, data)

# 设置标题和坐标轴标签
plt.title('Interval Bar Chart')
plt.xlabel('Interval')
plt.ylabel('Data')
#
# # 显示图表
# plt.show()

import matplotlib.pyplot as plt

def generate_bar_chart(intervals, data):
    # 创建柱状图
    plt.bar(intervals, data)

    # 设置标题和坐标轴标签
    plt.title('Interval Bar Chart')
    plt.xlabel('Interval')
    plt.ylabel('Data')

    # 显示图表
    plt.show()

# 调用函数生成柱状图
generate_bar_chart(intervals, data)


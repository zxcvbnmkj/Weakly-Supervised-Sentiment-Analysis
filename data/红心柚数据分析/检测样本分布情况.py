import pandas as pd
import seaborn
from matplotlib import pyplot as plt

df = pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\深度学习\dl_data\test_set.csv")

graph=seaborn.countplot(x='品质', data=df)
graph.bar_label(graph.containers[0])
# plt.savefig("完整测试集标签分布.jpg")
plt.show()
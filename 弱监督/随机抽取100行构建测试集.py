import pandas as pd
import seaborn
from matplotlib import pyplot as plt

# df = pd.read_csv(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\弱标签评论集_去除全2_修正粗负.csv')

#这个数据集是亲手更正的
df = pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\弱标签评论集_全2.csv")
sampled_data = df.sample(n=40,random_state=27)

droplist=sampled_data.axes[0]
print(droplist)


df2=df.drop(droplist)
print(len(df2))
df2.to_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\my_model训练数据\初始.csv",index=False)

# graph=seaborn.countplot(x='粗粒度', data=sampled_data)
# graph.bar_label(graph.containers[0])
# plt.show()

# sampled_data.to_csv("./all2_test.csv",index=False)




# sampled_data = df.sample(n=100,random_state=27)

# droplist=sampled_data.axes[0]
# print(droplist)
# graph=seaborn.countplot(x='粗粒度', data=sampled_data)
# graph.bar_label(graph.containers[0])
# plt.show()

# sampled_data.to_csv("./common_test100.csv",index=False)

# drop_index=[3870,  321, 6782, 3849, 4582, 3066, 4688, 2678, 1352, 6143,  221,
#             8230, 8634, 6949, 2687, 8695, 5098, 5108, 8984, 7257, 5900, 2289,
#             8191,  809,  353, 3693, 8255, 3942, 4677, 2239, 8078,  753, 4594,
#             1733, 7049, 3503, 7547, 8511, 1227,   54,  131, 2501, 1433, 4464,
#             6707, 1975, 6670, 5379, 2998, 7573, 1226, 8029, 8482, 2969, 1070,
#             5137, 8761, 2980, 8507, 6476,  380, 5921, 6457, 3353, 4112, 1843,
#             7620, 2584, 6994, 1349, 5409, 2629, 4816, 8515, 1464, 1146, 4136,
#             3003, 3119, 6432, 5785, 4997, 5593, 6003, 8879, 5682, 1997, 1008,
#             6384, 7909, 1761,  880,  391,  886, 2351, 5819, 2648, 8342, 3027,
#             4044]
#
# df=df.drop(drop_index)
#
# df.to_csv("./伪标签数据集.csv")
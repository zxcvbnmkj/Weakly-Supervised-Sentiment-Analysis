from matplotlib import pyplot as plt, lines

s1 = plt.scatter([0], [0], label='scatter', marker='+')
line1 = lines.Line2D([0], [0], label='line1', marker='o', lw=2, c='green')
# 方法2，注意别丢逗号
line2, = plt.plot([0], [0], label='line2', marker='X', lw=2, c='blue')

# 2、定义矩形图例
path1 = mpatches.Patch(color="red", label="patch")

handles = [line1, path1, line2, s1]
fig, ax = plt.subplots(figsize=(6.4, 0.32))  # 根据行数更改0.32
# ax.legend(handles=handles, labels=["a","b","c","d"], mode="expand", ncol = 4, borderaxespad = 0) # mode='expand', 水平展示图例
ax.legend(handles=handles, mode="expand", ncol=4, borderaxespad=0)
ax.axis("off")  # 去掉坐标刻度
plt.show()

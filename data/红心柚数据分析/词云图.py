import os.path
import random

import pandas as pd
from os import path
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator

# d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

#加载中文字体
font_path =r'D:\2filerj\anaconda3\envs\torch\Lib\site-packages\wordcloud\examples\fonts\SourceHanSerif\SourceHanSerifK-Light.otf'


#加载一张彩色图片
# img0=Image.open(path.join(d, "alice_color.png"))
# back_coloring = np.array(img0)
# display(img0)

#加载文本文件。分好词的。词与词、句与句之间都使用空格分隔
df=pd.read_csv(r"/2preprocess/分词后_主题建模.csv")
list=df["分词"].values
str=" ".join(list)


wc = WordCloud(font_path=font_path,background_color="white",max_words=300,width=800, height=400)

wc.generate(str)

img1=wc.to_image()
img1.save("./data/红心柚数据分析/词云图.jpg")
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 200)
#颜色蒙版
img2=wc.recolor(color_func=grey_color_func,random_state=1).to_image()

img2.show()
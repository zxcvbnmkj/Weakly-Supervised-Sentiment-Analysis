#用法说明
#https://www.51cto.com/article/679535.html
#https://github.com/MaartenGr/BERTopic
#因为无法下载multilingual的预训练词嵌入（文档2向量）模型
#所以在huggingface上手动下载了，更改了D:\2filerj\anaconda3\envs\torch\Lib\site-packages\bertopic\backen\_utils.py文件
#中的一些代码使其可以读取本地文件
#模型的本地存储路径是：D:\Pythonnnn\1resource\paraphrase-multilingual-MiniLM-L12-v2
#是fit_transform函数中的self.embedding_model = select_backend(self.embedding_model,
                                                  # language=self.language)
#select_backend函数出错
# -*- coding: utf-8 -*-d
from bertopic import BERTopic
import pandas as pd

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



df = pd.read_csv("D:\google_drive\weekly_learn\dl_data\训练集.csv", engine='python')
df2 = pd.read_csv("D:\google_drive\weekly_learn\dl_data\测试集.csv", engine='python')
df = pd.concat([df, df2], ignore_index=True)
# create model。nr_topics可以指定主题个数
# model = BERTopic(verbose=True,nr_topics=7,language='multilingual',embedding_model="SentenceTransformers")
#主题数(nr_topics)是一个指定主题数量的上限，实际主题数可能会因为算法收敛而少于nr_topics。language参数默认为英语，应该指定为多语言


# model = BERTopic(verbose=True,nr_topics=7,language='multilingual')

#当生成的主题数不够时（一直少于指定主题），加入n_gram_range=(1, 3)，这样可以使主题数变多（但也有副作用，会使得主题变乱，提取效果变差）


#！！！当nr_topics=7时，生成的主题一直是6个（0-5）.指定为8时才生成7个主题
# model = BERTopic(verbose=True,nr_topics=8,language='multilingual', n_gram_range=(1, 3))

model = BERTopic(verbose=True,nr_topics=10,language='multilingual', n_gram_range=(1, 3))


# model_path = "/path/to/your/model/directory/"
# # 加载模型
# model = BERTopic.load(model_path)



#convert to list
docs = df.评论.to_list()

topics, probabilities = model.fit_transform(docs)

# print(model.get_topic(1))

# model.visualize_topics()

# a=model.visualize_barchart()
# a.show()

# b=model.visualize_heatmap(top_n_topics=8)
b=model.visualize_heatmap(top_n_topics=10)

# b.write_html("./heatmap.html")
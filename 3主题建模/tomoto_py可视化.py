import pyLDAvis
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=Warning)
import pandas as pd
import tomotopy as tp



df=pd.read_csv(r"../2preprocess/分词后_主题建模.csv")

#使用PTModel
mdl = tp.PTModel(k=7, min_df=15, seed=555)
for words_str in df['分词']:
    words_list=words_str.split(" ")
    #确认words 是 非空词语列表
    if words_list:
        mdl.add_doc(words=words_list)

#训练
mdl.train(60)

#获取pyldavis需要的参数
topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq


prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=0, # tomotopy话题id从0开始，pyLDAvis话题id从1开始
    sort_topics=False #注意：否则pyLDAvis与tomotopy内的话题无法一一对应。
)


#可视化结果存到html文件中
pyLDAvis.save_html(prepared_data, 'ptmodel_60_7.html')
# -*- coding: utf-8 -*-

from skift import FirstColFtClassifier
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


df = pandas.DataFrame([['woof', 0], ['meow', 1]], columns=['txt', 'lbl'])


"""sk中机器学习函数使用方法。X和y必须是数字"""
sk_clf = svm.SVC(probability=True)
#将文本转换为向量
vectorizer = TfidfVectorizer()
#train_vec是词向量
train_vec = vectorizer.fit_transform(df['txt'])

sk_clf.fit(train_vec, df['lbl'])
print(sk_clf.predict(train_vec[0]))
print(sk_clf.predict_proba(train_vec[0]))

"""skift中X不能是向量，只要是汉字就行（因为FT模型是通过CBOW训练的，不需要额外嵌入）
！！！X的df中有两个[]。双层的。不加报错"""
# sk_clf = FirstColFtClassifier(lr=0.3, epoch=10)
# sk_clf.fit(df[['txt']], df['lbl'])
# print(sk_clf.predict([['woof']]))
# print(sk_clf.predict_proba([['woof']]))


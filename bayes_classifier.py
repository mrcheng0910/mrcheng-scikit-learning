# encoding:utf-8
"""
使用bayes分类器对数据进行预测分析，最基础内容研究。
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


## 获取数据
news = fetch_20newsgroups(subset='all')
print len(news.data)
# print news.data[0]

## 分割数据
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

## 文本特征向量转换模块
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)  # 标准化
X_test = vec.transform(X_test)     # why
#
mnb = MultinomialNB()  # 初始化
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)

print mnb.score(X_test,y_test)
print classification_report(y_test, y_predict,target_names=news.target_names)




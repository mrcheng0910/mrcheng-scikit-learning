# encoding:utf-8
"""
使用svm分类器对数据进行预测分析，最基础内容研究。
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

## 获取数据
digits = load_digits()
print digits.data.shape

## 分割数据
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

## 标准化处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

## 分类
lsvc = LinearSVC()  # 初始化
lsvc.fit(X_train,y_train)    # 训练模型
y_predict = lsvc.predict(X_test)  # 预测

## 性能测评
print lsvc.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))

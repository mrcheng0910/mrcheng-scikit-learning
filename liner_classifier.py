# encoding:utf-8
"""
使用线性分类器对数据进行预测分析，最基础内容研究。
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report


## 1. 数据获取和处理
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('data.txt',names=column_names)  # 读取数据并为每列命名
data = data.replace(to_replace='?',value=np.nan)  # 将？替换为标准缺失值
data = data.dropna(how='any')    # 丢弃带有缺失值的数据（只要有一个维度丢失就丢去）

## 2. 训练数据和测试数据
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)

print y_train.value_counts()  # 查看y值分布
print y_test.value_counts()

## 3. 数据标准化(方差为1，均值为0)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

## 4. 分类
lr = LogisticRegression()  # 初始化分类器
sgdc = SGDClassifier()     # 初始化分类器

lr.fit(X_train,y_train)    # 训练模型
lr_y_predict = lr.predict(X_test)  # 预测测试数据，得到结果

sgdc.fit(X_train,y_train)   # 训练模型
sgdc_y_predict = sgdc.predict(X_test)  # 预测测试数据，得到结果

## 5. 性能分析
print "accuracy of LR Classifier:", lr.score(X_test, y_test)   # LR分类器自带评分函数
print classification_report(y_test,lr_y_predict,target_names=['Benign', 'Malignant'])  # 使用report模块进行评分

print "accuarcy of SGD Classifier",sgdc.score(X_test,y_test)  # 自带函数
print classification_report(y_test,sgdc_y_predict,target_names=['Benign', 'Malignant'])

# encoding:utf-8
"""
最简单的使用svm分类iris花
"""

from sklearn import datasets
from sklearn import svm
import cPickle as pickle


clf = svm.SVC()   # svm 分类器(classifier)
iris = datasets.load_iris()  # 加载iris数据
X, y = iris.data, iris.target  # 训练集和结果集
clf.fit(X, y)   # 训练模型

print clf.predict(X[0:1])  # 预测结果，注意使用X[0:1]这种形式，[[,,,,]],而不是X[0],[,,,,]
print y[0]           # 实际结果

# 训练模型持久存储（字符串），可以使用dump保存为文本，load进行加载
s = pickle.dumps(clf)  # 保存训练模型
clf = pickle.loads(s)  # 加载训练模型

print clf.predict(X[0:2])
print y[0:2]
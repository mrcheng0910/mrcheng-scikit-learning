# encoding:utf-8
"""
使用svm简单分类iris花
"""

from sklearn import datasets
from sklearn import svm
import cPickle as pickle
from sklearn.externals import joblib

clf = svm.SVC()   # svm 分类器(classifier)
iris = datasets.load_iris()  # 加载iris数据
X, y = iris.data, iris.target  # 训练集和结果集
clf.fit(X, y)   # 训练模型
print clf.predict(X[0:1])  # 预测结果，注意使用X[0:1]这种形式，[[,,,,]],而不是X[0],[,,,,]
print y[0]           # 实际结果

# 使用pickle将训练模型持久存储（字符串），可以使用dump保存为文本，load进行加载
s = pickle.dumps(clf)  # 保存训练模型
clf = pickle.loads(s)  # 加载训练模型
print clf.predict(X[0:2])
print y[0:2]

# 使用joblib将训练模型持久存储，当数据较大时候，可以使用，多使用这个
joblib.dump(clf, 'test.pkl')
clf = joblib.load('test.pkl')
print clf.predict(X[0:2])



# encoding:utf-8
"""
使用kn分类器对数据进行预测分析，最基础内容研究。
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


## 获取数据
iris = load_iris()
print iris.data.shape
print iris.data[0]
# print iris.DESCR   # 查看数据描述

## 分割数据
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

## 文本特征向量转换模块
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 标准化
X_test = ss.transform(X_test)     # why
#
knc = KNeighborsClassifier()  # 初始化
knc.fit(X_train,y_train)
y_predict = knc.predict(X_test)

print knc.score(X_test,y_test)
print classification_report(y_test, y_predict,target_names=iris.target_names)




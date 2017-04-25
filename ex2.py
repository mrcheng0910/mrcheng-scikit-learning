#encoding: utf-8
"""
多标签进行分类
"""
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print classif.fit(X, y).predict(X)

y = LabelBinarizer().fit_transform(y)
print y
print classif.fit(X, y).predict(X)
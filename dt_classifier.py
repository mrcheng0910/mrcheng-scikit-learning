# encoding:utf-8
"""
使用kn分类器对数据进行预测分析，最基础内容研究。
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import GradientBoostingClassifier  #梯度提升决策树


## 获取数据
titanic = pd.read_csv('titanic.txt')
# print titanic.head()
# print titanic.info()

## 数据处理
X = titanic[['pclass','age','sex']]
y = titanic['survived']

# print X.info()   # 发现部分数据缺失

X['age'].fillna(X['age'].mean(), inplace=True)

# print X.info()

## 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

## 特征转换
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='records'))
# print vec.feature_names_
X_test = vec.transform(X_test.to_dict(orient='records'))

## 单一决策树分类和性能
dtc = DecisionTreeClassifier()   #初始化
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)
print dtc.score(X_test, y_test)
print classification_report(y_predict,y_test,target_names=['died','survived'])

## 随机森林分类和性能
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)
print rfc.score(X_test,y_test)
print classification_report(rfc_y_pred,y_test)

## 梯度提升决策树分类和性能
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)
print gbc.score(X_test,y_test)
print classification_report(gbc_y_pred,y_test)



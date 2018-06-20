#!/usr/bin/python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics  import  accuracy_score

#loading iris
iris=load_iris()

#traning flowers.features is stored in iris.data
#output accordingly is stored in iris.target

#now splitting into test and train data sets

train_iris,test_iris,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)

#calling knn algo
knnclf=KNeighborsClassifier(n_neighbors=3)

#calling dsc algo
dsclf=tree.DecisionTreeClassifier()

#data training
knntrained=knnclf.fit(train_iris,train_target)
dsctrained=dsclf.fit(train_iris,train_target)

#testing algo
#predicted output
knnoutput=knntrained.predict(test_iris)
print(knnoutput)
dscoutput=knntrained.predict(test_iris)
print(dscoutput)

#original output
print(test_target)

#calculating accuracy
knnpct=accuracy_score(test_target,knnoutput)
print(knnpct)
dscpct=accuracy_score(test_target,dscoutput)
print(dscpct)

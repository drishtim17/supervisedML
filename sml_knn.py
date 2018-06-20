#!/usr/bin/python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics  import  accuracy_score

#loading iris
iris=load_iris()

#traning flowers.features is stored in iris.data
#output accordingly is stored in iris.target

#now splitting into test and train data sets

train_iris,test_iris,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.1)

#calling knn algo
knnclf=KNeighborsClassifier(n_neighbors=5)

#data training
knntrained=knnclf.fit(train_iris,train_target)

#testing algo
#predicted output
output=knntrained.predict(test_iris)
print(output)

#original output
print(test_target)

#calculating accuracy
pct=accuracy_score(test_target,output)
print(pct)

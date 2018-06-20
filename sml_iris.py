#!/usr/bin/python3

import sklearn
import numpy
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#loadig all data
iris=load_iris()

print(dir(iris))  #directory of all iris

print(iris.target_names) #types of iris

print(iris.feature_names) #features names sl,sw,pl,pw

#print(iris.data) #data of features

#print(iris.data.shape) #(rows,cloumn)

#print(iris.target.shape) # shape of target that is rows 150

#print(iris.target) #target data in 0,1,2 fo6rm

#only setosa
setosa=iris.data[0:50]

#only setosa data
s_data=iris.target[0:50]

print(s_data)

print(s_data.size)

#to delete the data
x=[0,50,100]

#target data value after deleting
only_target_training=numpy.delete(iris.target,x)
print("only_target_trainig :")
print(only_target_training)
print(only_target_training.size)

#data value after deleting
only_data_training=numpy.delete(iris.data,x,axis=0)
print("only_data_trainig :")
print(only_data_training)

#testing target
test_target=iris.target[x]
print("only_target_test :")
print(test_target)

test_data=iris.data[x]
print("only_data_test :")
print(test_data)

#calling algorithm
clf=tree.DecisionTreeClassifier()
trained=clf.fit(only_data_training,only_target_training)
output=trained.predict(test_data)
print(output)

x=[iris.feature_names]
y=[iris.target_names]
plt.xlabel("feature")
plt.xlabel("target")
plt.show()


#!/usr/bin/python3
import sklearn
from sklearn.datasets import load_iris

#loading iris
iris=load_iris()

#traning flowers.features is stored in iris.data
#output accordingly is stored in iris.target

#now splitting into test and train data sets

from sklearn.model_selection import train_test_split

x,y,z,a=train_test_split(iris.data,iris.target,test_size=0.1)

'''
here
x is train_iris {all features values conating 99%}
y is remainig test_iris {10%of features}
z is train_target {all labels containing 90% of iris.target}
a is test_target {remaining 10% of iris.target}
'''
print("value of x :")
print(x)
print("value of y :")
print(y)
print("value of z :")
print(z)
print("value of a :")
print(a)

#calling decision algorithm classifier
from sklearn import tree

dsclf=tree.DecisionTreeClassifier()

#now taring data with decision
trained=dsclf.fit(x,z)#(train_iris,train_target)

#time for prediction
output=trained.predict(y)#test_iris
print("output :")
print(output)

#acutal output
print("Actual output :")
print(z)#test_target

#  now calculating  accuracy 
from  sklearn.metrics  import  accuracy_score

check_pct=accuracy_score(a,output)#(test_target,test_iris(output))
print("Accuracy :")
print(check_pct)

#exporting graphs for decision tree
   
#export_graphviz(decision_tree, out_file="tree.dot", max_depth=None, feature_names=None, class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3)
print("treee")
tree.export_graphviz(dsclf, out_file="tree.dot", max_depth=7, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
 



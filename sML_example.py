#!/usr/bin/python3

import sklearn
from sklearn import tree

#features about apple and orange

data=[[100,0],[130,0],[135,1],[150,1]]

output=["apple","apple","orange","orange"]

#decision tree algorithm call

algo=tree.DecisionTreeClassifier()

#train data
trained_algo=algo.fit(data,output)

#now testing phase
predict=trained_algo.predict([[136,0]])

#printing output
print(predict)

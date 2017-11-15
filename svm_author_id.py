#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
"""
# using 1% of training set
a = int(len(features_train)/100)
b = int(len(labels_train)/100)
features_train = features_train[:a]
labels_train = labels_train[:b]
"""

#clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print("accuracy =", accuracy)

print ("answer to element 10 is = ", pred[10])
print ("answer to element 26 is = ", pred[26])
print ("answer to element 50 is = ", pred[50])

a=0
b=0
print("length =", len(pred))
for ii in range(len(pred)):
   if(pred[ii] == 1): a += 1
   else: b += 1 

print( "number of prediction for Chris = ",a)
print( "number of prediction for Sara = ",b)

#########################################################



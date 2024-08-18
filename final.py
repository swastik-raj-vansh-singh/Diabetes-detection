# -*- coding: utf-8 -*-
"""
@author: swastik

"""

#CLASSIFICATION MODELS
import pandas as pd

#IMPORTING THE PIMA INDIAN DIABETES DATASET
dataset= pd.read_csv("diabetes.csv")

X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8:9].values

#TAKING CARE OF MISSING DATA
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0,strategy="mean", axis=0)
imputer = imputer.fit(X[:,0:8])
X[:,0:8] = imputer.transform(X[:,0:8])

#SPLITTING DATA INTO TRAING SET AND TEST SET
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#FITTING OUR MODEL TO VARIOUS CLASSIFICATION MODELS

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logistic_regressor  = LogisticRegression()
logistic_regressor.fit(X_train,y_train)
print()
print()
print("Accuracy Logistic regression",logistic_regressor.score(X_test,y_test))


#NEAREST NEIGHBOUR 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)
print("Accuracy K Nearest Neighbours",knn.score(X_test,y_test))


#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
SVM = SVC(kernel ='linear',random_state=0)
SVM.fit(X_train,y_train)
print("Accuracy Linear SVM",SVM.score(X_test,y_test))

#RBF KERNEL SVM
rbf_SVM = SVC(kernel="rbf",random_state=0)
rbf_SVM.fit(X_train,y_train)
print("Accuracy rbf SVM",rbf_SVM.score(X_test,y_test))


#NAIVE BAYES CLASSIFIACTION 
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_test,y_test)
print("Accuracy naives bayes",naive_bayes.score(X_test,y_test))


#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion="entropy",random_state=0)
decision_tree.fit(X_train,y_train)
print("Accuracy Decision tree",decision_tree.score(X_test,y_test))


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
random_forest.fit(X_train,y_train)
print("Accuracy Random Forest",random_forest.score(X_test,y_test))


#VOTING CLASSIFIER
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('lr',logistic_regressor),('dt',naive_bayes),('SVM',rbf_SVM)],voting='hard')
voting.fit(X_train,y_train)
print("Accuracy voting ",voting.score(X_test,y_test))






print("FEATURE EXTRACTION DONE")
#FEATURE EXTRACTION
X= dataset.iloc[:,[1,5,6,7]].values
y = dataset.iloc[:,8:9].values

#TAKING CARE OF MISSING DATA
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0,strategy="mean", axis=0)
imputer = imputer.fit(X[:,0:8])
X[:,0:8] = imputer.transform(X[:,0:8])

#SPLITTING DATA INTO TRAING SET AND TEST SET
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#FITTING OUR MODEL TO VARIOUS CLASSIFICATION MODELS

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logistic_regressor  = LogisticRegression()
logistic_regressor.fit(X_train,y_train)
print()
print()
print("Accuracy Logistic regression",logistic_regressor.score(X_test,y_test))


#NEAREST NEIGHBOUR 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)
print("Accuracy K Nearest Neighbours",knn.score(X_test,y_test))


#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
SVM = SVC(kernel ='linear',random_state=0)
SVM.fit(X_train,y_train)
print("Accuracy Linear SVM",SVM.score(X_test,y_test))

#RBF KERNEL SVM
rbf_SVM = SVC(kernel="rbf",random_state=0)
rbf_SVM.fit(X_train,y_train)
print("Accuracy rbf SVM",rbf_SVM.score(X_test,y_test))


#NAIVE BAYES CLASSIFIACTION 
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_test,y_test)
print("Accuracy naives bayes",naive_bayes.score(X_test,y_test))


#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion="entropy",random_state=0)
decision_tree.fit(X_train,y_train)
print("Accuracy Decision tree",decision_tree.score(X_test,y_test))


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
random_forest.fit(X_train,y_train)
print("Accuracy Random Forest",random_forest.score(X_test,y_test))


#VOTING CLASSIFIER
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('lr',logistic_regressor),('dt',naive_bayes),('SVM',rbf_SVM)],voting='hard',weights=[1,1,2])
voting.fit(X_train,y_train)
print("Accuracy voting ",voting.score(X_test,y_test))



from tkinter import *
from functools import partial  
values=[0,0,0,0,0,0,0,0,0,0]
root = Tk()
label1= Label(root,text="Enter the values of the following to check whether the person is diabetic or not")
label1.grid(row=0)

e2=StringVar()
e3=StringVar()
e4=StringVar()
e5=StringVar()
e6=StringVar()
e7=StringVar()
e8=StringVar()
e9=StringVar()


label2=Label(root,text="Preganancies:")
label2.grid(row=1)
entry2=Entry(root,textvariable=e2)
entry2.grid(row=1,column=1)

label3=Label(root,text="Glucose:")
label3.grid(row=3)
entry3=Entry(root,textvariable=e3)
entry3.grid(row=3,column=1)

label4=Label(root,text="Blood pressure")
label4.grid(row=5)
entry4=Entry(root,textvariable=e4)
entry4.grid(row=5,column=1)

label5=Label(root,text="Skin thickness:")
label5.grid(row=7)
entry5=Entry(root,textvariable=e5)
entry5.grid(row=7,column=1)

label6=Label(root,text="insulin:")
label6.grid(row=9)
entry6=Entry(root,textvariable=e6)
entry6.grid(row=9,column=1)

label7=Label(root,text="BMI")
label7.grid(row=11)
entry7=Entry(root,textvariable=e7)
entry7.grid(row=11,column=1)

label8=Label(root,text="Diabetes Pedigree Function:")
label8.grid(row=13)
entry8=Entry(root,textvariable=e8)
entry8.grid(row=13,column=1)


label9=Label(root,text="Age:")
label9.grid(row=15)
entry9=Entry(root,textvariable=e9)
entry9.grid(row=15,column=1)


def onclick(e2,e3,e4,e5,e6,e7,e8,e9):
    p1=e2.get()
    values[0]=float(p1)
    p2=e3.get()
    values[1]=float(p2)
    p3=e4.get()
    values[2]=float(p3)
    p4=e5.get()
    values[3]=float(p4)
    p5=e6.get()
    values[4]=float(p5)
    p6=e7.get()
    values[5]=float(p6)
    p7=e8.get()
    values[6]=float(p7)
    p8=e9.get()
    values[7]=float(p8)
    return
    
onclick = partial(onclick,e2,e3,e4,e5,e6,e7,e8,e9)
button = Button(root,text="SUBMIT",command=onclick)
button.grid(row=17,column=1)
root.mainloop()

v=[[values[1],values[5],values[6],values[7]]]
y_pred=voting.predict(v)
print(y_pred)

if(y_pred==0):
    print("The Patient is Diabetic")
else:
    print("The Patient is NOT Diabetic")

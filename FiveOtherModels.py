# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:53:10 2020

@author: chambm6
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:15:02 2020

@author: chambm6
"""
#Import necessary packages
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import np_utils
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/drug_consumption.data.csv")
#Filtering out participants who answered yes to using fictional drug semer
fake = df[df['semer']!='CL0']
df = df.drop([df.index[727], df.index[817], df.index[1516], df.index[1533], df.index[1698], df.index[1769], df.index[1806], df.index[1823]])

#List of features
feature_col_names = ['age','gender', 'education', 'country','ethnicity','nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']

#List of drug types
columns = ['alcohol','amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
           'ecstasy', 'heroin', 'ketamine', 'legalh', 'LSD', 'meth', 'mushrooms','nicotine', 'semer', 'VSA']

#Transforming target variable class names to integers
for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

#Drug we are testing for classification, this changes for each classification run
predicted_class_names = ['VSA']
#Creating atribute and target variable columns
X = df[feature_col_names]
y = df[predicted_class_names]
#Binarizing the target variable
dummy_y = np_utils.to_categorical(y)

#Furth splitting the training data into a training and validation set used to validate the model
xtrain, X_test, ytrain, y_test = train_test_split(X, y,  test_size=0.20, random_state=42)

'''
clf = CatBoostClassifier(iterations = 6000, depth = 5, learning_rate = .05, loss_function = "MultiClass")
#Fitting model to training data
clf.fit(xtrain,ytrain)
pred = clf.predict(X_test)

#Calculating testing accuracy
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))
'''
'''
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
X_test = scaler.transform(X_test)
'''
'''
clf = RandomForestClassifier(n_estimators = 50,max_depth = 2, criterion = 'entropy', random_state = 42)
clf.fit(xtrain, ytrain)
pred = clf.predict(X_test)

#Calculating testing accuracy
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))
'''
'''
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
#indices = np.argsort(importances)[::-1]

feat = []
imp = []

for f in range(xtrain.shape[1]):
    feat.append(feature_col_names[f])
    imp.append(importances[f])
    
bar = pd.DataFrame()
#Only plotting top 10 importance features out of 160
bar['Feature'] = feat[:12]
bar['Importance'] = imp[:12]
bar.plot(x="Feature", y="Importance", kind="bar", legend = None)

plt.bar([x for x in range(len(importances))], importances, color = 'steelblue')
plt.xticks(np.arange(12),labels= feature_col_names, rotation = 90, fontsize = 18)
plt.yticks(fontsize=18)
plt.xlabel('')
plt.show()
'''

'''
clf = XGBClassifier(objective = "multi:softmax", learning_rate = .1, n_estimtors = 500, max_depth = 5, num_class = 7)
#mlb = MultiLabelBinarizer()
#y = mlb.fit_transform(ytrain)
clf.fit(xtrain,ytrain)
#Predicting class labels
pred = clf.predict(X_test)
#Calculating testing accuracy
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))
'''
'''
from sklearn.svm import SVC 
clf = SVC(kernel = 'poly',degree = 3, C = 1).fit(xtrain, ytrain) 
pred = clf.predict(X_test) 
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))
'''
'''
#Building KNN Classifier
clf = KNeighborsClassifier(n_neighbors = 60, weights = 'distance')
#Fitting model on training data
clf.fit(xtrain,ytrain)
#Predicting class labels
pred = clf.predict(X_test)

#Calculating testing accuracy
accu = metrics.accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accu * 100.0))
'''


from mlxtend.evaluation import feature_importance_permutation

import seaborn as sns
sns.set_style("darkgrid")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Calculating and graphing feature importance
results = permutation_importance(clf, xtrain, ytrain, scoring='accuracy', n_repeats = 5)

# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance

plt.bar([x for x in range(len(importance))], importance, color = 'steelblue')
plt.xticks(np.arange(12),labels= feature_col_names, rotation = 90, fontsize = 18)
plt.yticks(fontsize=18)
plt.show()

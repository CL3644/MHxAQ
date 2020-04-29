import csv 
import sys
import operator
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
import numpy as np
from numpy import random
from numpy.random import randn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import seaborn as sns
import collections

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import tree, svm
# import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score


###########
# CustomScaler(): https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
# user: @J_C

# normalize(df): https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame/55645516
# user: @Amirhos Imani
###########


class CustomScaler(BaseEstimator,TransformerMixin): 
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.ix[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


def normalize(df):
	result = df.copy()
	for feature_name in df.columns:
		if "Concentration" in feature_name:
			max_value = df[feature_name].max()
			min_value = df[feature_name].min()
		else:
			max_value =  1
			min_value = 0
		result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	return result

##################################
# Preprocessing

PATH = 'data.csv'
data =  pd.read_csv(PATH)

y =  data.columns[:-1]
x = data.columns[-1]

target = "Mental Illness"
variables = ["Sex", "Concentration", "Veteran", "random"]


z = data[target].values.reshape(-1, 1)

X = data

for label in data.columns:
	print(label)
	y=0
	for item in variables:
		if item in label:
			y = 1
	if y ==  0:
		X = X.drop(label, 1)

toscale = []
notscale = []

for col in X.columns:
	if "Concentration" in col:
		toscale.append(col)
	else:
		notscale.append(col)

print(toscale)
print(notscale)

scale = CustomScaler(toscale)
X = scale.fit_transform(X)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, random_state = 42)

######################################################
# PCA for training all variables

# pca = PCA(0.95)
# X_train = pca.fit_transform(X_train)
# X_train = pd.DataFrame(data = X_train)

# X_test = pca.transform(X_test)

# print(pca.explained_variance_ratio_)

######################################################
# FIND BEST PARAMETERS FOR DECISION TREE USING RANDOMIZED SEARCH

# crit = ["gini", "entropy"]
# splitter = ["best", "random"]
# md = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# min_samples_split = [2, 20, 200]
# min_samples_leaf = [1, 10, 100, 1000]

# random_grid = {'criterion': crit, 'splitter': splitter, 'max_depth': md, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# print(random_grid)

# rst_dt = DecisionTreeClassifier()
# rf_random = RandomizedSearchCV(estimator = rst_dt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, z_train)
# print(rf_random.best_params_)

# dtc = DecisionTreeClassifier(min_samples_split = 2, splitter = 'random', criterion = 'gini', max_depth = 4, min_samples_leaf = 1)
# ret = dtc.fit(X_train, z_train)
# z_predictions = dtc.predict(X_test)

# score = f1_score(z_predictions, z_test)

# print("DTC f1 score: " + str(score))

# for name, importance in zip(X_train.columns, dtc.feature_importances_):
# 		print(name, importance)

# input()

##########################################
# SVM WORK IN PROGRESS

# model_svm = svm.SVC(class_weight = 'balanced')

# C = [0.001, 0.01, 0.1, 1, 10]
# gamma = [0.001, 0.01, 0.1, 1]
# kernel = ['linear', 'poly', 'rbf']
# degree = [2, 3]
# param_grid = {'C': C, 'gamma':gamma}
# grid_search = GridSearchCV(model_svm, param_grid, cv = 3, n_jobs = -1)
# grid_search.fit(X_train, z_train)
# print(grid_search.best_params_)

# input()

# clf = model_svm
# clf.fit(X_train, z_train)
# pred = clf.predict(X_test)
# acc = accuracy_score(z_test, pred)
# score = f1_score(z_test, pred)

# print("f1 score: " + str(score))
# print("DTC mean accuracy: " + str(acc))

# input()


######################################################
# Decision Trees

max_depths = np.linspace(1, 15, 15, endpoint=True)

train_results = []
test_results = []
accuracy_results = []

for d in max_depths:
	print("max depth: " + str(d))
	print("\n")

	dtc =  DecisionTreeClassifier(max_depth = d, random_state = 1, class_weight = "balanced")
	ret = dtc.fit(X_train, z_train)
	z_predictions = dtc.predict(X_test)
	print(collections.Counter(z_predictions))

	score = f1_score(z_test, z_predictions)
	acc = accuracy_score(z_test, z_predictions)

	print("DTC f1 score: " + str(score))
	print("DTC mean accuracy: " + str(acc))
	test_results.append(score)
	accuracy_results.append(acc)

	for name, importance in zip(X_train.columns, dtc.feature_importances_):
		print(name, importance)

	print("\n\n")
	print("-----------------")

line2, = plt.plot(max_depths, test_results, 'r', label="DT f1 score")
line1, =  plt.plot(max_depths, accuracy_results, 'orange', label="DT mean accuracy")

#######################################################
# Random forest 

max_depths = np.linspace(1, 15, 15, endpoint=True)
print(max_depths)

train_results = []
test_results = []
rf_acc = []
for d in max_depths:
	print("max depth: " + str(d))
	print("\n")
	rf = RandomForestClassifier(100, max_depth = d, class_weight = "balanced")

	# rf.fit(X_train, z_train.ravel())
	rf.fit(X_train, z_train)
	z_predictions_rf = rf.predict(X_test)

	print(collections.Counter(z_predictions_rf))

	score = f1_score(z_test, z_predictions_rf)
	acc = accuracy_score(z_test, z_predictions_rf)
	print("RF f1 score: " + str(score))
	print("RF accuracy: " + str(acc))
	test_results.append(score)
	rf_acc.append(acc)

	for name, importance in zip(X_train.columns, rf.feature_importances_):
		print(name, importance)

	print("\n\n")
	print("-----------------")

line1, = plt.plot(max_depths, test_results, 'b', label="RF F1 Score")
line4, = plt.plot(max_depths, rf_acc, 'green', label="rf acc")

plt.ylabel('f1 score')
plt.xlabel('tree depth')
plt.show()







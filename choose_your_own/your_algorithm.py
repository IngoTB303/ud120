#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
def initplot():
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
	plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
	plt.legend()
	plt.xlabel("bumpiness")
	plt.ylabel("grade")
	plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

def adaboost(features_train, labels_train, features_test, labels_test):
	from sklearn.ensemble import AdaBoostClassifier
	adaclf = AdaBoostClassifier(n_estimators=100,learning_rate=1)
	adaclf.fit(features_train, labels_train)
	acc = adaclf.score(features_test, labels_test)
	print "accuracy AdaBoostClassifier: ", acc
	plt.title('AdaBoostClassifier')
	return adaclf

def knearest(features_train, labels_train, features_test, labels_test):
	from sklearn.neighbors import KNeighborsClassifier
	kclf = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto',
		p=3, metric='minkowski', n_jobs=1)
	kclf.fit(features_train, labels_train)
	acc = kclf.score(features_test, labels_test)
	print "accuracy NearestNeighbors: ", acc
	plt.title('k-nearest')
	return kclf	

def randomforest(features_train, labels_train, features_test, labels_test):
	from sklearn.ensemble import RandomForestClassifier
	rfclf = RandomForestClassifier(n_estimators=5, criterion='gini', 
		min_samples_split=4, min_samples_leaf=4)
	rfclf.fit(features_train, labels_train)
	acc = rfclf.score(features_test, labels_test)
	print "accuracy RandomForestClassifier: ", acc
	plt.title('RandomForestClassifier')
	return rfclf	

try:
    #clf = adaboost(features_train, labels_train, features_test, labels_test)
    #prettyPicture(clf, features_test, labels_test)
    #plt.show()
    # BEST ACCURACY:
    clf = knearest(features_train, labels_train, features_test, labels_test)
    prettyPicture(clf, features_test, labels_test)
    plt.show()
    #clf = randomforest(features_train, labels_train, features_test, labels_test)
    #prettyPicture(clf, features_test, labels_test)
    #plt.show()
except NameError:
    pass

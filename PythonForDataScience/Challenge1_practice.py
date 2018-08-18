from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np 

# Data and labels
X = [[181, 80, 47], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
	 [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
	 [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
	 'female', 'male', 'male']

# RandomForestClassifier, LogisticRegression, GaussianNB
# using the default values for all the hyperoarameters
clf_rfc = RandomForestClassifier()
clf_lr = LogisticRegression()
clf_gnb = GaussianNB()

# Training the models
clf_rfc.fit(X, Y)
clf_lr.fit(X, Y)
clf_gnb.fit(X, Y)

# Testing using the same Data
pred_rfc = clf_rfc.predict(X)
acc_rfc = accuracy_score(Y, pred_rfc)
print ('Accuracy for RandomForestClassifier: {}'.format(acc_rfc))

pred_lr = clf_lr.predict(X)
acc_lr = accuracy_score(Y, pred_lr)
print ('Accuracy for LogisticRegreesion: {}'.format(acc_lr))

pred_gnb = clf_gnb.predict(X)
acc_gnb = accuracy_score(Y, pred_gnb)
print ('Accuracy for GaussianNB: {}'.format(acc_gnb))

# The best classifier from rfc, lr, acc_gnb
index = np.argmax([acc_rfc, acc_lr, acc_gnb])
classifiers = {0: 'RFC', 1: 'LR', 2: 'GNB'}
print ('Best gender classifier is {}'.format(classifiers[index]))
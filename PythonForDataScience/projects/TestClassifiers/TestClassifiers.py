import unittest


class TestClassifiers(unittest.TestCase):

	def test_decision_tree(self):
		from sklearn import tree
		#[height, weight, shoe_size]
		X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     		 [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42],
     		 [181, 85, 43]]
		Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X, Y)

		prediction = clf.predict([[190, 70, 43]])
		print (prediction)

	def test_random_forest(self):
		from sklearn.ensemble import RandomForestClassifier
		#[height, weight, shoe_size]
		X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
		Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

		clf = RandomForestClassifier(n_estimators=2)
		clf = clf.fit(X, Y)

		prediction = clf.predict([[190, 70, 43]])
		print (prediction)

	def test_k_nearest_neighbour(self):
		from sklearn.neighbors import KNeighborsClassifier
		#[height, weight, shoe_size]
		X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
		Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(X, Y)

		prediction = neigh.predict([[190, 70, 43]])
		print (prediction)

	def test_logistic_regression(self):
		from sklearn.linear_model import LogisticRegression
    	#[height, weight, shoe_size]
		X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
		Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

		neigh = LogisticRegression()
		neigh.fit(X, Y)

		prediction = neigh.predict([[190, 70, 43]])
		print (prediction)

	def test_naive_bayes(self):
		from sklearn.naive_bayes import GaussianNB
		gnb = GaussianNB()
		X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
		Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

		gnb = gnb.fit(X, Y)

		prediction = gnb.predict([[190, 70, 43]])
		print (prediction)


test = TestClassifiers()
test.test_decision_tree()
test.test_random_forest()
test.test_k_nearest_neighbour()
test.test_logistic_regression()
test.test_naive_bayes()
#coding=utf-8
#author：xubaochuan
#date：2016/09/25

from sklearn import preprocessing
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

attributes = {
	0:["?"],
	1:["?","Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
	2:["?"],
	3:["?","Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
	4:["?"],
	5:["?","Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
	6:["?","Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
	7:["?","Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
	8:["?","White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
	9:["?","Female", "Male"],
	10:["?"],
	11:["?"],
	12:["?"],	
	13:["?","United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
	14:[">50K", "<=50K"],
	}

def load(datapath):
	fr = open(datapath)
	line = fr.readline().strip()
	example_col = len(line.split('\t'))
	data = fr.readlines()
	
	example_list = []
	for line in data:
		line = line.strip()
		line_list = line.split(',')
		if line == '':
			continue
		for index,val in enumerate(line_list):
			if val == "?":
				line_list[index] = 0
				continue
			if len(attributes[index]) > 1:
				line_list[index] = attributes[index].index(val.strip())
			else:
				line_list[index] = val.strip()
		example_list.append(line_list)
	examples = np.array(example_list,dtype=np.float64)
#	examples = preprocessing.scale(examples)
	min_max_scaler = preprocessing.MinMaxScaler()
	examples = min_max_scaler.fit_transform(examples)
	return examples

def linear_svc(train, test):
	svc = svm.SVC(kernel = 'linear')
	svc.fit(train[:,:-1], train[:,-1:].reshape((train[:,-1:].shape[0],)))
	score = svc.score(test[:,:-1], test[:,-1:].reshape((test[:,-1:].shape[0],)))
	print("svc for adult classification: %f" % score)

def gnb(train, test):
	gnb = GaussianNB()
	gnb.fit(train[:,:-1], train[:,-1:].reshape((train[:,-1:].shape[0],)))
	score = gnb.score(test[:,:-1], test[:,-1:].reshape((test[:,-1:].shape[0],)))
#	print gnb.predict([[0.45205479,0.25,0.0482376,0,0.8,0.,0.35714286,0.4,0.,1., 0.,0., 0.12244898,0.02439024]])
	print("gnb for adult classification: %f" % score)

def knn(train, test):
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(train[:,:-1], train[:,-1:].reshape((train[:,-1:].shape[0],)))
	score = knn.score(test[:,:-1], test[:,-1:].reshape((test[:,-1:].shape[0],)))
	print("knn for adult classification: %f" % score)

def dt(train, test):
	dt = DecisionTreeClassifier()
	dt.fit(train[:,:-1], train[:,-1:].reshape((train[:,-1:].shape[0],)))
	score = dt.score(test[:,:-1], test[:,-1:].reshape((test[:,-1:].shape[0],)))
	print("dt for adult classification: %f" % score)

if __name__=='__main__':
	example_train = load("adult.data.txt")
	example_test = load("adult.test.txt")
	perm = np.random.permutation(example_train.shape[0])
	example_train = example_train[perm]
	linear_svc(example_train, example_test)
	gnb(example_train, example_test)
	knn(example_train, example_test)
	dt(example_train, example_test)

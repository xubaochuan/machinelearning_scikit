#coding=utf-8

import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def svr(x, y, prediction):
	model = svm.SVR()
	model.fit(x, y)
	output = model.predict(prediction)
	return output

def lr(x, y, prediction):
	model = LinearRegression()
	model.fit(x, y)
	output = model.predict(prediction)
	return output

def rfr(x, y, prediction):
	model = LinearRegression()
	model.fit(x, y)
	output = model.predict(prediction)
	return output

#coding=utf-8
from sklearn import svm
from sklearn.linear_model import LinearRegression
import os
import numpy as np
from sklearn import preprocessing
import csv
from sklearn.ensemble import RandomForestRegressor

prediction_list_week = [[3,15,0,67,7924],
		[3,15,1,65,7691],
		[3,15,2,61,7514],
		[3,15,3,63,7518],
		[3,15,4,62,7442],
		[3,15,5,63,7535],
		[3,16,0,66,5615],
		[3,16,1,62,5399],
		[3,16,2,64,5541],
		[3,16,3,67,5613],
		[3,16,4,64,5457],
		[3,16,5,68,5596],
		[3,17,0,66,3562],
		[3,17,1,64,3339],
		[3,17,2,63,3269],
		[3,17,3,63,3212],
		[3,17,4,60,3151],
		[3,17,5,62,3207]]

prediction_list = [[15,0,67,7924],
		[15,1,65,7691],
		[15,2,61,7514],
		[15,3,63,7518],
		[15,4,62,7442],
		[15,5,63,7535],
		[16,0,66,5615],
		[16,1,62,5399],
		[16,2,64,5541],
		[16,3,67,5613],
		[16,4,64,5457],
		[16,5,68,5596],
		[17,0,66,3562],
		[17,1,64,3339],
		[17,2,63,3269],
		[17,3,63,3212],
		[17,4,60,3151],
		[17,5,62,3207]]

def svr(x, y):
	model = svm.SVR()
	model.fit(x, y)
	min_max_scaler = preprocessing.MinMaxScaler()
	normal_prediction_list = min_max_scaler.fit_transform(prediction_list)
#	normal_prediction_list = prediction_list
	result = model.predict(normal_prediction_list)
	contents = []
	for i in range(18):
		datestr = '2016-09-14-' + str(prediction_list[i][1]) + '-' + str(prediction_list[i][2])
		number = result[i]
		contents.append([number, datestr])
	return contents


def lr(x, y):
	model = LinearRegression()
	model.fit(x, y)
	min_max_scaler = preprocessing.MinMaxScaler()
	normal_prediction_list = min_max_scaler.fit_transform(prediction_list)
#	normal_prediction_list = prediction_list
	result = model.predict(normal_prediction_list)
	contents = []
	for i in range(18):
		datestr = '2016-09-14-' + str(prediction_list[i][1]) + '-' + str(prediction_list[i][2])
		number = result[i]
		contents.append([number, datestr])
	return contents

def rfr(x, y):
	model = RandomForestRegressor()
	model.fit(x, y)
	min_max_scaler = preprocessing.MinMaxScaler()
	normal_prediction_list = min_max_scaler.fit_transform(prediction_list)
#	normal_prediction_list = prediction_list
	result = model.predict(normal_prediction_list)
	contents = []
	for i in range(18):
		datestr = '2016-09-14-' + str(prediction_list[i][1]) + '-' + str(prediction_list[i][2])
		number = result[i]
		contents.append([number, datestr])
	return contents

if __name__=='__main__':
	filedir = './processeddata/step2/'
	result_path = './processeddata/prediction/result.csv'
	target_file = open(result_path, "wb")
	open_file_object = csv.writer(target_file)
	open_file_object.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])
	aps = os.listdir(filedir)
	for ap in aps:
		print ap
		examples_list = []
		fr = open(filedir + ap)
		datas = fr.readlines()
		for line in datas:
			line_list = line.strip().split(',')
			examples_list.append(line_list)
		examples = np.array(examples_list, dtype=np.float32)
		y = examples[:,-1]
#		x = examples[:,:-1]
		min_max_scaler = preprocessing.MinMaxScaler()
		x = min_max_scaler.fit_transform(examples[:,1:-1])
		result = rfr(x, y)
		for i in result:
			open_file_object.writerow([round(i[0], 1), ap, i[1]])
	target_file.close()
	print "success"

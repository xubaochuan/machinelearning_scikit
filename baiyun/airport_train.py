import os
import csv
import numpy as np
from sklearn import preprocessing
import pandas as pd
import classifier
import evaluate
prediction_list = [[3,12,0,72,9028],
		[3,12,1,68,8249],
		[3,12,2,70,8463],
		[3,12,3,67,8435],
		[3,12,4,66,8126],
		[3,12,5,67,8301],
		[3,13,0,73,9607],
		[3,13,1,67,8841],
		[3,13,2,67,9017],
		[3,13,3,68,9320],
		[3,13,4,69,9015],
		[3,13,5,67,8913],
		[3,14,0,65,8827],
		[3,14,1,64,8830],
		[3,14,2,61,8282],
		[3,14,3,64,9106],
		[3,14,4,66,8862],
		[3,14,5,65,8878]]

def output_format(output, examples):
	contents = []
	for i in range(18):
		mean = cal_passengers_mean(examples, prediction_list[i][1], prediction_list[i][2])
		value = punish(mean, output[i])
		datestr = '2016-09-14-' + str(prediction_list[i][1]) + '-' + str(prediction_list[i][2])
		contents.append([value, datestr])
	return contents

def punish(mean, current):
	measure_rate = (mean - current)/(mean + 0.01)
	if measure_rate > 0.8:
		value = mean
	elif measure_rate > 0.5:
		value = current + (mean - current)*0.8
	else:
		value = mean
	return value

def cal_passengers_mean(examples, hour, period):
	examples = pd.DataFrame(examples, columns=list('ABCDEF'))
	query = examples[(examples['B']==hour) & (examples['C']==period)]['F']
	mean =  query.sum(axis=0)/query.shape[0]
	return mean

def normalize(train, prediction):
	train_length = train.shape[0]
	merge_array = np.concatenate((train, prediction))
	min_max_scaler = preprocessing.MinMaxScaler()
	merge_array = min_max_scaler.fit_transform(merge_array)
	train = merge_array[:train_length,:]
	prediction = merge_array[train_length:,:]
	return train,prediction

if __name__=='__main__':
	trainset_dir = './train/trainset/'
	result_path = './train/result.csv'
	target_file = open(result_path, "wb")
	open_file_object = csv.writer(target_file)
	open_file_object.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])
	aps = os.listdir(trainset_dir)
	for ap in aps:
		print ap
		examples_list = []
		fr = open(trainset_dir + ap)
		datas = fr.readlines()
		for line in datas:
			line_list = line.strip().split(',')
			examples_list.append(line_list)
		examples = np.array(examples_list, dtype=np.float32)
		prediction_nparray = np.array(prediction_list, dtype=np.float32)
		x, prediction = normalize(examples[:,1:-1], prediction_nparray[:,1:])
#		x = examples[:,1:-1]
#		prediction = prediction_nparray[:,1:]
		y = examples[:,-1]
		output = classifier.svr(x, y, prediction)
		result = output_format(output, examples)
		for i in result:
			open_file_object.writerow([round(i[0], 2), ap, i[1]])
	target_file.close()
	print "success"
	evaluate.score('./train/result.csv', './train/evaluation.csv')

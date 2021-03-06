import os
import csv
import numpy as np
from sklearn import preprocessing
import classifier
import pandas as pd

prediction_list = [[3,15,0,67,7924],
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
		value = current + (mean - current)
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
	trainset_dir = './prediction/trainset/'
	result_path = './prediction/result.csv'
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
		y = examples[:,-1]
		output = classifier.svr(x, y, prediction)
		result = output_format(output, examples)
		for i in result:
			open_file_object.writerow([round(i[0], 2), ap, i[1]])
	target_file.close()
	print "success"

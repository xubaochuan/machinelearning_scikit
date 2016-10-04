#coding=utf-8
import pandas as pd
import airport_train

def score(prediction_file, result_file):
	prediction = pd.read_csv(prediction_file)
	result = pd.read_csv(result_file)
	test_score = 0	
	for index,row in result.iterrows():
		query = prediction[(prediction["WIFIAPTag"]==row[1]) & (prediction["slice10min"]==row[2])]
		test_score += (row[0] - query.values[0,0])**2
	print test_score

if __name__=='__main__':
	score('./train/result.csv', './train/evaluation.csv')

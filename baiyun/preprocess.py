#coding=utf-8
import os
import time
import datetime
import pandas as pd
import csv

position_list = ['E1', 'E2', 'E3', 'EC', 'T1', 'W1', 'W2', 'W3', 'WC']

#calculate the flight time and the passenger number
def flight_info(filepath, targetpath):
	flight_departures = pd.read_csv(filepath)
	if os.path.exists(targetpath):
		os.remove(targetpath)
	target_file = open(targetpath, "wb")
	open_file_object = csv.writer(target_file)
#	open_file_object.writerow(['flight_id', 'week', 'hour', 'period', 'num'])
	for name,group in flight_departures.groupby(["flight_ID"]):
		print name
		time_dict = {}
		for name2,group2 in group.groupby(["flight_time"]):
			if name2 == '':
				continue
			flight_time =  datetime.datetime.strptime(name2, "%Y/%m/%d %H:%M:%S")
			week = datetime.datetime(flight_time.year, flight_time.month, flight_time.day).strftime("%w")
			hour = flight_time.hour
			period = flight_time.minute/10
			if time_dict.has_key(week):
				time_dict[week]['num'] += group2.shape[0]
				if group2.shape[0]>time_dict[week]['max']:
					time_dict[week]['max'] = group2.shape[0]
					time_dict[week]['hour'] = hour
					time_dict[week]['period'] = period
			else:
				time_dict[week] = {}
				time_dict[week]['num'] = group2.shape[0]
				time_dict[week]['max'] = group2.shape[0]
				time_dict[week]['hour'] = hour
				time_dict[week]['period'] = period
		for key,val in time_dict.items():
			print name, key, val['hour'], val['period'], val['num']
			open_file_object.writerow([name, key, val['hour'], val['period'], val['num']])
	target_file.close()

#get the flight number and the passenger number for the waiting time
def flight_count(flight_info, cal_time):
	array = cal_time.split(',')
	week = int(array[0])
	hour = int(array[1])
	period = int(array[2])
	flights = 0
	passengers = 0
	for row in flight_info:
		if (row[1] == week):
			d_hour = row[2] - hour
			d_period = row[3] - period
			if d_hour >= 0:
				if (d_hour <2) or (d_hour == 2 and d_period ==0 ):
					flights += 1
					passengers += row[4]
	return flights, passengers

#split the origin data into files which named by wifi ap name
def ap_split(filepath, targetdir):
	fr = open(filepath)
	contents = fr.readlines()
	for line in contents:
		line = line.strip()
		position = line[0:2]
		if position in position_list:
			line_list = line.split(',')
			day_of_week = datetime.datetime(int(line_list[2][0:4]), int(line_list[2][5:7]), int(line_list[2][8:10])).strftime("%w")
			hour = int(line_list[2][11:13])
			period = int(line_list[2][14:16])/10
			write_data = str(day_of_week) + ',' + str(hour) + ',' + str(period) + ',' + line_list[1] + "\n"
			fw = open(targetdir + line_list[0], "a")
			fw.write(write_data)
			fw.close()
			print line_list[0]
	fr.close()

#generate the train set
def period_mean(filepath, targetdir, flight_info_path):
	flight_info_fr = open(flight_info_path)
	flight_info_contents = flight_info_fr.readlines()
	flight_info = []
	for line in flight_info_contents:
		line_list = line.strip().split(',')
		for index in range(1,5):
			line_list[index] = int(line_list[index])
		flight_info.append(line_list)
	file_list = os.listdir(filepath)
	for ap in file_list:
		ap_list = []
		ap_dict = {}
		fr = open(filepath + ap)
		contents = fr.readlines()
		for line in contents:
			if line != '':
				line_list = line.split(',')
				if int(line_list[0])==6:
					continue
				dict_key = line_list[0] + ',' + line_list[1] + ',' + line_list[2]
				if ap_dict.has_key(dict_key):
					ap_dict[dict_key].append(float(line_list[3]))
				else:
					ap_list.append(dict_key)
					ap_dict[dict_key] = [float(line_list[3])]
		fr.close()
		for key in ap_list:
			val = ap_dict[key]
			flights, passengers = flight_count(flight_info, key)
			mean = sum(val)/len(val)
			write_data = key + "," + str(flights) + "," + str(passengers) + "," + str(mean) + "\n"
			fw = open(targetdir + ap, "a")
			fw.write(write_data)
			fw.close()
		print ap
	print "success"

#generate the prediction feature
def generate_prediction_csv(flight_info_path, target_path):
	target_file = open(target_path, "wb")
	open_file_object = csv.writer(target_file)
	
	flight_info_fr = open(flight_info_path)
	flight_info_contents = flight_info_fr.readlines()
	flight_info = []
	for line in flight_info_contents:
		line_list = line.strip().split(',')
		for index in range(1,5):
			line_list[index] = int(line_list[index])
		flight_info.append(line_list)
	week = 2
	for i in range(12,15):
		for j in range(0,6):
			timestr = str(week) + ',' + str(i) + ',' + str(j)
			flights,passengers = flight_count(flight_info, timestr)
			open_file_object.writerow([week, i, j, flights, passengers])
	target_file.close()

def generate_evaluation_data(from_path, to_path, test_file_path):
	test_file = open(test_file_path, "wb")
	test_file_object = csv.writer(test_file)
	test_file_object.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])

	file_list = os.listdir(from_path)
	for ap in file_list:
		to_file = open(to_path + ap, "wb")
		to_file_object = csv.writer(to_file)		

		fr = open(from_path + ap)
		lines = fr.readlines()
		for line in lines:
			if line == '':
				continue			
			line_list = line.strip().split(',')
			if int(line_list[0]) == 3 and int(line_list[1])>=12 and int(line_list[2])<=14:
				timestr = '2016-09-14-' + line_list[1] + '-' + line_list[2]
				test_file_object.writerow([line_list[5], ap, timestr])
			else:
				to_file_object.writerow([line_list[0],line_list[1],line_list[2],line_list[3],line_list[4],line_list[5]])
		to_file.close()
	test_file.close()


if __name__=="__main__":
#	flight_info("./origindata/airport_gz_departure_chusai_1stround.csv", "./processeddata/flight/flight_info.csv")
#	ap_split("./origindata/WIFI_AP_Passenger_Records_chusai_1stround.csv", "./processeddata/step1/")
#	period_mean("./processeddata/step1/", "./processeddata/step2/", "./processeddata/flight/flight_info.csv")
#	generate_prediction_csv("./processeddata/flight/flight_info.csv", "./processeddata/prediction/date_train.csv")
	generate_evaluation_data('./processeddata/step2/', './processeddata/train/', './processeddata/test.csv')

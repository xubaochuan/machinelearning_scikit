import tensorflow as tf
import os
import numpy as np
from sklearn import preprocessing

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
test = np.array(prediction_list, dtype=np.float32)
min_max_scaler = preprocessing.MinMaxScaler()
test = min_max_scaler.fit_transform(test)
examples_list = []
fr = open("./processeddata/step2/E1-1A-2<E1-1-02>")
datas = fr.readlines()
for line in datas:
	line_list = line.strip().split(',')
	examples_list.append(line_list)
	examples = np.array(examples_list, dtype=np.float32)
	train_y = examples[:,-1:]
#	x = examples[:,:-1]
	min_max_scaler = preprocessing.MinMaxScaler()
	train_x = min_max_scaler.fit_transform(examples[:,:-1])

x = tf.placeholder(tf.float32, [None,5])

W1 = tf.Variable(tf.ones([5,80]))
b1 = tf.Variable(tf.ones([80]))

#h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h1 = tf.matmul(x, W1) + b1

W2 = tf.Variable(tf.ones([80,1]))
b2 = tf.Variable(tf.ones([1]))

y = tf.matmul(h1, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.reduce_sum((y_-y)**2)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()

sess.run(init)

batch_count = 25
batch_size = 20
index = 0
for i in range(100):
	for i in range(batch_count):
		start = i * batch_size
		end = i * batch_size + batch_size
		print W1.eval()
#		print cross_entropy.eval(feed_dict={x:train_x, y_:train_y})
#		print y.eval(feed_dict={x:train_x[start:end], y_:train_y[start:end]})
		sess.run(train_step, feed_dict={x:train_x[start:end], y_:train_y[start:end]})
		index += 1
		if index==3:
			exit()
prediction = y.eval(feed_dict={x:test})
print prediction
sess.close()

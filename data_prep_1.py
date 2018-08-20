import tensorflow as tf
from tensorflow.keras.layers import Conv2D , Dense , Flatten, MaxPooling2D , Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical 
import random
import sys
import matplotlib.pyplot as plt


def data_process():
	data = pd.read_csv('data/fer2013.csv')
	emotion = ['Angry', 'Disgust', 'Fear', 'Happy',
			   'Sad', 'Surprise', 'Neutral']


	# Removing disgust  from training set
	print(data.shape)
	data= data.drop(data[data['emotion']==1].index) 
	print(data.shape)


	# In[4]:


	nemotion = ['Angry', 'Fear', 'Happy',
			   'Sad', 'Surprise', 'Neutral']
	ndict={i:e for i , e in (enumerate(nemotion))}
	# ndict


	# In[6]:


	tdata = data[data.Usage.values == 'Training']
	print(tdata.emotion.value_counts())
	ind = tdata.emotion.value_counts().index
	print(ind)
	print(min(tdata.emotion.value_counts()))
	print(tdata.shape)
	train = pd.DataFrame({data.columns[0]:[] , data.columns[1]:[] , data.columns[2]:[]})
	for i in ind:
		temp = tdata[tdata.emotion.values == i].sample(min(tdata.emotion.value_counts()))
		train = train.append(temp)
	print(train.emotion.value_counts())
	print(train.shape)


	# In[11]:


	data = data.drop(data[data['Usage'] == 'Training'].index)
	print(data.shape)
	data = pd.concat([data, train])
	print(data.shape)
	count = (data.Usage.value_counts())
	data.index = range(0,data.shape[0])
	# data


	Y = data.emotion.values
	Y
	for i in range(0,Y.shape[0]):
		if(Y[i]>1):
			Y[i] = Y[i]-1
	max(Y)
	data['emotion']= Y
	# data.head()


	X = data.values
	pixels = X[:,1]
	print(type(pixels))
	print(len(pixels[0]))
	print(pixels.shape)
	X = np.zeros((pixels.shape[0], 48*48))

	for ix in range(X.shape[0]):
		p = pixels[ix].split(' ')
		for iy in range(X.shape[1]):
			X[ix, iy] = int(p[iy])
			
	X = X.reshape(X.shape[0], 48 , 48)
	from sklearn.utils import shuffle
	X, Y = shuffle(X, Y, random_state=0)

	x_train = np.zeros((count['Training'],48,48))
	y_train = np.zeros((count['Training']))
	j = 0
	for i in data.index:
	#     print(data['Usage'][i])
	#     break
		if(data['Usage'][i] == 'Training'):
			y_train[j] = Y[i]
			x_train[j] = X[i]
			j = j+1

	x_cv = np.zeros((count['PublicTest'],48,48))
	y_cv = np.zeros((count['PublicTest']))
	j = 0 
	for i in range(0,data.shape[0]):
		if(data['Usage'][i] == 'PublicTest'):
			y_cv[j] = (Y[i])
			x_cv[j] =(X[i])
			j = j +1

	x_test = np.zeros((count['PrivateTest'],48,48))
	y_test = np.zeros((count['PrivateTest']))
	j = 0
	for i in range(0,data.shape[0]):
		if(data['Usage'][i] == 'PrivateTest'):
			y_test[j] = Y[i]
			x_test[j] = X[i]
			j = j +1
			
	print(x_train.shape, y_train.shape , x_test.shape , y_test.shape, x_cv.shape , y_cv.shape)
		
	x_train = x_train.reshape(x_train.shape[0] , 48,48 , 1)
	x_cv = x_cv.reshape(x_cv.shape[0] , 48,48 , 1)
	x_test = x_test.reshape(x_test.shape[0] , 48,48, 1)
	
	num_classes = 6
	y_train = tf.keras.utils.to_categorical(y_train , num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)
	y_cv = tf.keras.utils.to_categorical(y_cv , num_classes)
	
	print(x_train.shape, y_train.shape , x_test.shape , y_test.shape, x_cv.shape , y_cv.shape)



	# print(x_train.shape, y_train.shape , x_test.shape , y_test.shape, x_cv.shape , y_cv.shape)
	return x_train, y_train , x_test , y_test, x_cv , y_cv

if __name__=='__main__':
	data_process()
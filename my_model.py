
import tensorflow as tf
from tensorflow.keras.layers import Conv2D , Dense , Flatten, MaxPooling2D , Dropout , BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical 
import random
import sys
import matplotlib.pyplot as plt



def My_Model(weights_path = None):

	model = tf.keras.models.Sequential()

	model.add(Conv2D(filters=32 , kernel_size=(3,3) , activation ='relu' , padding='same',input_shape=(48,48,1)))
	model.add(Conv2D(filters=32 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=64 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(Conv2D(filters=64 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Dropout(0.2))

	model.add(Conv2D(filters=128 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(Conv2D(filters=128, kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(Conv2D(filters=256, kernel_size=(3,3) , activation ='relu' , padding='same'))
	# model.add(Dropout(0.2))
	model.add(Conv2D(filters=256, kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=512 , kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3,3) , activation ='relu' , padding='same'))
	# model.add(Conv2D(filters=512, kernel_size=(3,3) , activation ='relu' , padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	
	model.add(Dense(513,activation='relu'))
	# model.add(Dropout(0.3))
	model.add(Dense(513,activation='relu'))
	# model.add(Dense(512,activation='relu'))
	# model.add(Dropout(0.4))
	model.add(Dense(6,activation='softmax'))

	if weights_path is not None:
		model.load_weights(weights_path)


	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
				  loss=tf.keras.losses.categorical_crossentropy,
				  metrics=['accuracy'])
	import os
	os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	from keras.utils.vis_utils import plot_model
	plot_model(model, to_file='model.png')
	# from IPython.display import SVG
	# from keras.utils.vis_utils import model_to_dot

	# SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
	print('Model Created') 
	print(model.summary())	
	return model
	
if __name__=='__main__':
	My_Model(weights_path=None)
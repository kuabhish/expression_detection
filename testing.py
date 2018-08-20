import cv2
import matplotlib.pyplot as plt
import cv2
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
import data_prep_1
import numpy as np



cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('my_demo_1.avi',fourcc, 20.0, (640,480))
import my_model
model = my_model.My_Model(weights_path='expression_cnn_weights.h5')
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# print(frame.shape)
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.rectangle(frame,(200,100),(450,400),(0,255,0),3)
	roi = gray[100:400 , 200:450]
	roi = cv2.resize(roi , (48,48))	
	# print(roi.shape)
	roi = roi.reshape(1,48,48,1)
	# print(roi.shape)
	answer = model.predict(roi)
	# print(answer , answer.shape)
	nemotion = ['Angry', 'Fear', 'Happy',
			   'Sad', 'Surprise', 'Neutral']
	ndict={i:e for i , e in (enumerate(nemotion))}
	print(ndict)
	print(np.argmax(np.array(answer)))
	cv2.putText(frame , ndict[np.argmax(np.array(answer))],org= (200,100), fontFace =cv2.FONT_HERSHEY_SIMPLEX ,
			fontScale = 1 ,color =(255,0,0) )
	# cv2.imshow('roi', roi)
	# frame = cv2.resize(frame , dsize = (100,100))
	# print(frame.shape)
	out.write(frame)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
# img = cv2.imread('1.jpg')
# cv2.imshow('Image', img)
# cv2.waitKey(0)

cv2.destroyAllWindows()
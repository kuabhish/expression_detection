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
import my_model
import cv2

def train(model,x_train, y_train , x_test , y_test, x_cv , y_cv):

	callbacks = [
		EarlyStopping(patience=4,monitor='val_acc')
	]


	# In[ ]:


	history = model.fit(x_train, y_train , epochs = 50, 
					   callbacks = callbacks , batch_size = 64 , 
					   validation_data = (x_cv, y_cv)) #10 epochs for time .... too long


	# In[ ]:


	# model.save('expression_cnn_model.h5')
	model.save_weights('expression_cnn_weights.h5', save_format='h5')

	fig, ax = plt.subplots(2,1)
	ax[0].plot(history.history['loss'], color='b', label="Training loss")
	ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
	legend = ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
	ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
	legend = ax[1].legend(loc='best', shadow=True)
	plt.show()


	# In[ ]:


	# pred = model.predict(x_test)
	# score = model.evaluate(x_test , y_test )
	# print('loss=' , score[0])
	# print('accuracy=',  score[1])
	# import numpy as np
	# index = 102
	# print('prediction:', (np.argmax((np.array(pred[index])))) , '\nactual:', np.argmax((np.array(y_test[index]))))
	# new1 = np.squeeze(x_test[index])
	# print(new1.shape)
	# cv2.imwrite('messigray.png',new1)
	# new=cv2.resize(new1,(200,200))/255.0
	# plt.imshow(new , cmap = 'gray')
	# plt.show()
	# cv2.imshow('Image', new)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
# if __name__=='__main__':
	# train(my_model.My_model(weights_path=None), )

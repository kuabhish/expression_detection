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

x_train, y_train , x_test , y_test, x_cv , y_cv= data_prep_1.data_process()
print('done')
print(x_train.shape, y_train.shape, x_cv.shape , y_cv.shape)

import my_model
model = my_model.My_Model(weights_path=None)

import train
train.train(model , x_train, y_train , x_test , y_test, x_cv , y_cv)

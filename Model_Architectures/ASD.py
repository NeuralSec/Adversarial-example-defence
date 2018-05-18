import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

global nb_classes
nb_classes = 10

def ASD(In_Shape):
	asd_in = Input(shape=In_Shape)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Flatten()(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	asd_out = Dense(nb_classes, activation='softmax')(x)
	asd = Model(asd_in,asd_out)
	return asd

def Normal_loss_function(y_true, y_pred):
	#loss = -K.log(y_true - y_pred)
	#loss = K.mse()
	loss = K.categorical_crossentropy(y_true, y_pred)
	return loss

def Adv_loss_function(y_true, y_pred):
	loss = -K.categorical_crossentropy(y_true, y_pred)
	return loss

def ASD_Multiinput(In_Shape):
	Normal_input = Input(shape=In_Shape, name='normal_input')
	Adv_input = Input(shape=In_Shape, name='adv_input')
	x = keras.layers.concatenate([Normal_input, Adv_input])
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Flatten()(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	normal_output = Dense(nb_classes, activation='softmax')(x)
	adv_output = Dense(nb_classes, activation='softmax')(x)
	model = Model(inputs=[Normal_input, Adv_input], outputs=[normal_output, adv_output])
	model.compile(optimizer='adam',
				  loss={'normal_output': Normal_loss_function, 'adv_output': Adv_loss_function},
				  loss_weights={'normal_output': 0.7, 'adv_output': 0.3})

def Vulnerability_capturer(In_Shape):
	g_input = Input(shape=In_Shape)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = Flatten()(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	g_out = Dense(nb_classes, activation='softmax')(x)
	generator = Model(g_input,g_out)
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
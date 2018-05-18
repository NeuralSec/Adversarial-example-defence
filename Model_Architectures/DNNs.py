import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Reshape, Activation, Flatten, Activation, Multiply, BatchNormalization
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

nb_classes = 10
EPOCHS = 100
VALID_SPLIT = 0.1
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


class CNN:
	def __init__(self, In_Shape):
		inputs = Input(shape=In_Shape, name='Normal_inputs')
		x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='C1')(inputs)
		x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='C2')(x)
		x = MaxPooling2D(pool_size=(2,2), name='MP1')(x)
		x = Dropout(0.2, name='Drop1')(x)
		x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C3')(x)
		x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C4')(x)
		x = MaxPooling2D(pool_size=(2,2), name='MP2')(x)
		x = Dropout(0.2, name='Drop2')(x)
		x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='C5')(x)
		x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='C6')(x)
		x = MaxPooling2D(pool_size=(2,2), name='MP3')(x)
		x = Dropout(0.2, name='Drop3')(x)
		x = Flatten(name='Flatten')(x)
		x = Dense(512, activation='relu', name='D1')(x)
		x = Dropout(0.2, name='Drop4')(x)
		x = Dense(nb_classes)(x)
		outputs = Activation('softmax', name='normal_output')(x)
		#outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
		self.model = Model(inputs=inputs, outputs=outputs)
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	def train(self, sess, train_x, train_y):
		self.sess = sess
		self.model.fit(train_x, train_y, epochs=EPOCHS, validation_split=VALID_SPLIT, callbacks=[earlystopping])
		return self.model


class MPN:
	def __init__(self, In_Shape):
		inputs = Input(shape=In_Shape, name='Normal_inputs')
		x = Dense(100)(inputs)
		x = Dense(100)(x)
		x = Dense(nb_classes)(x)
		outputs = Activation('softmax', name='normal_output')(x)
		self.model = Model(inputs=inputs, outputs=outputs)
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	def train(self, sess, train_x, train_y):
		self.sess = sess
		self.model.fit(train_x, train_y, epochs=EPOCHS, validation_split=VALID_SPLIT, callbacks=[earlystopping])
		return self.model


class Autoencoder:
	def __init__(self, In_Shape):
		inputs = Input(shape=In_Shape)
		x = Convolution2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(inputs)
		x = AveragePooling2D(pool_size=(2,2))(x)
		x = Convolution2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
		x = UpSampling2D(size=(2, 2))(x)
		x = Convolution2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
		decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
		self.model = Model(inputs=inputs, outputs=decoded)
		self.model.compile(optimizer='adam', loss='mean_squared_error', 
					       metrics=['categorical_accuracy'])
	def train(self, sess, train_x, train_y, EPOCHS=EPOCHS):
		self.sess = sess
		self.model.fit(train_x, train_y, epochs=EPOCHS, validation_split=VALID_SPLIT)
		return self.model
	def single_batch_train(self, sess, train_x, train_y):
		self.sess = sess
		self.model.train_on_batch(train_x, train_y)


class Upsampling_generator():
	def __init__(self, In_Shape):
		self.dim = 7
		self.depth = 64+64+64+64
		self.dropout=0.4
		inputs = Input(shape=In_Shape)
		x = Dense(self.dim*self.dim*self.depth)(inputs)
		#x = BatchNormalization(momentum=0.9)(x)
		x = Activation('relu')(x)
		x = Reshape((self.dim, self.dim, self.depth))(x)
		x = Dropout(self.dropout)(x)
		x = UpSampling2D()(x)
		x = Conv2DTranspose(int(self.depth/2), 5, padding='same')(x)
		#x = BatchNormalization(momentum=0.9)(x)
		x = Activation('relu')(x)
		x = UpSampling2D()(x)
		x = Conv2DTranspose(int(self.depth/4), 5, padding='same')(x)
		#x = BatchNormalization(momentum=0.9)(x)
		x = Activation('relu')(x)
		x = Conv2DTranspose(int(self.depth/8), 5, padding='same')(x)
		#x = BatchNormalization(momentum=0.9)(x)
		x = Activation('relu')(x)
		x = Conv2DTranspose(1, 5, padding='same')(x)
		decoded = Activation('sigmoid')(x)
		self.model = Model(inputs=inputs, outputs=decoded)
		self.model.compile(optimizer='adam', loss='mean_squared_error', 
					       metrics=['categorical_accuracy'])
	def train(self, sess, train_x, train_y, EPOCHS=EPOCHS):
		self.sess = sess
		self.model.fit(train_x, train_y, epochs=EPOCHS, validation_split=VALID_SPLIT)
		return self.model
	def single_batch_train(self, sess, train_x, train_y):
		self.sess = sess
		self.model.train_on_batch(train_x, train_y)
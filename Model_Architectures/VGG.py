import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten, Activation, Multiply
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import SGD

global nb_classes
nb_classes = 10

def VGG_16(In_Shape):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=In_Shape))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def MNIST_Substitute(In_Shape):
	inputs = Input(shape=In_Shape, name='Normal_inputs')
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='C1')(inputs)
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='C2')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP1')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C3')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C4')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP2')(x)
	x = Flatten(name='Flatten')(x)
	x = Dense(200, activation='relu', name='D1')(x)
	x = Dense(200, activation='relu', name='D2')(x)
	#x = Dense(nb_classes)(x)
	#outputs = Activation('softmax', name='normal_output')(x)
	#outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
	model = Model(inputs=inputs, outputs=x)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])
	return model

def CIFAR_Substitute(In_Shape):
	inputs = Input(shape=In_Shape, name='Normal_inputs')
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C1')(inputs)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='C2')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP1')(x)
	x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='C3')(x)
	x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='C4')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP2')(x)
	x = Flatten(name='Flatten')(x)
	x = Dense(256, activation='relu', name='D1')(x)
	x = Dense(256, activation='relu', name='D2')(x)
	#x = Dense(nb_classes)(x)
	#outputs = Activation('softmax', name='normal_output')(x)
	#outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
	model = Model(inputs=inputs, outputs=x)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])
	return model

def Custom(In_Shape):
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
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])
	return model

def Mnist_Custom(In_Shape):
	inputs = Input(shape=In_Shape, name='Normal_inputs')
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', name='C1')(inputs)
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', name='C2')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP1')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', name='C3')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', name='C4')(x)
	x = MaxPooling2D(pool_size=(2,2), name='MP2')(x)
	x = Flatten(name='Flatten')(x)
	x = Dense(200, activation='relu', name='D1')(x)
	x = Dense(200, activation='relu', name='D2')(x)
	outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])
	return model

def Custom_Adv(In_Shape):
	#Define shared layers
	Shared_C1 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C2 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP1 = MaxPooling2D(pool_size=(2,2))
	Shared_DP1 = Dropout(0.2, name='Drop1')
	Shared_C3 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C4 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP2 = MaxPooling2D(pool_size=(2,2))
	Shared_DP2 = Dropout(0.2)
	Shared_C5 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C6 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP3 = MaxPooling2D(pool_size=(2,2))
	Shared_DP3 = Dropout(0.2)
	Shared_Flatten = Flatten()
	Shared_Dense = Dense(512, activation='relu')
	Shared_DP4 = Dropout(0.2)
	
	#Built sequential model
	Normal_inputs = Input(shape=In_Shape, name='Normal_inputs')
	Normal_C1 = Shared_C1(Normal_inputs)
	Normal_C2 = Shared_C2(Normal_C1)
	Normal_MP1 = Shared_MP1(Normal_C2)
	Normal_DP1 = Shared_DP1(Normal_MP1)
	Normal_C3 = Shared_C3(Normal_DP1)
	Normal_C4 = Shared_C4(Normal_C3)
	Normal_MP2 = Shared_MP2(Normal_C4)
	Normal_DP2 = Shared_DP2(Normal_MP2)
	Normal_C5 = Shared_C5(Normal_DP2)
	Normal_C6 = Shared_C6(Normal_C5)
	Normal_MP3 = Shared_MP3(Normal_C6)
	Normal_DP3 = Shared_DP3(Normal_MP3)
	Normal_Flatten = Shared_Flatten(Normal_DP3)
	Normal_Dense = Shared_Dense(Normal_Flatten)
	Normal_DP4 = Shared_DP4(Normal_Dense)

	Adversarial_inputs = Input(shape=In_Shape, name='Adversarial_inputs')
	Adversarial_C1 = Shared_C1(Adversarial_inputs)
	Adversarial_C2 = Shared_C2(Adversarial_C1)
	Adversarial_MP1 = Shared_MP1(Adversarial_C2)
	Adversarial_DP1 = Shared_DP1(Adversarial_MP1)
	Adversarial_C3 = Shared_C3(Adversarial_DP1)
	Adversarial_C4 = Shared_C4(Adversarial_C3)
	Adversarial_MP2 = Shared_MP2(Adversarial_C4)
	Adversarial_DP2 = Shared_DP2(Adversarial_MP2)
	Adversarial_C5 = Shared_C5(Adversarial_DP2)
	Adversarial_C6 = Shared_C6(Adversarial_C5)
	Adversarial_MP3 = Shared_MP3(Adversarial_C6)
	Adversarial_DP3 = Shared_DP3(Adversarial_MP3)
	Adversarial_Flatten = Shared_Flatten(Adversarial_DP3)
	Adversarial_Dense = Shared_Dense(Adversarial_Flatten)
	Adversarial_DP4 = Shared_DP4(Adversarial_Dense)
	
	normal_output = Dense(nb_classes, activation='softmax', name='normal_output')(Normal_DP4)
	adversarial_output = Dense(nb_classes, activation='softmax', name='adversarial_output')(Adversarial_DP4)

	model = Model(inputs=[Normal_inputs,Adversarial_inputs], outputs=[normal_output, adversarial_output])
	return model

def Custom_Reg(In_Shape):
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
	Normal_outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
	Reg_outputs = Dense(nb_classes, activation='softmax', name='regularised_output')(x)
	model = Model(inputs=inputs, outputs=[Normal_outputs, Reg_outputs])
	return model

def Custom_Reg_Super(In_Shape):
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
	Normal_outputs = Dense(nb_classes, activation='softmax', name='normal_output')(x)
	Reg_outputs0 = Dense(nb_classes, activation='softmax', name='regularised_output0')(x)
	Reg_outputs1 = Dense(nb_classes, activation='softmax', name='regularised_output1')(x)
	Reg_outputs2 = Dense(nb_classes, activation='softmax', name='regularised_output2')(x)
	Reg_outputs3 = Dense(nb_classes, activation='softmax', name='regularised_output3')(x)
	Reg_outputs4 = Dense(nb_classes, activation='softmax', name='regularised_output4')(x)
	Reg_outputs5 = Dense(nb_classes, activation='softmax', name='regularised_output5')(x)
	Reg_outputs6 = Dense(nb_classes, activation='softmax', name='regularised_output6')(x)
	Reg_outputs7 = Dense(nb_classes, activation='softmax', name='regularised_output7')(x)
	Reg_outputs8 = Dense(nb_classes, activation='softmax', name='regularised_output8')(x)
	Reg_outputs9 = Dense(nb_classes, activation='softmax', name='regularised_output9')(x)
	model = Model(inputs=inputs, outputs=[Normal_outputs,
										  Reg_outputs0,
										  Reg_outputs1,
										  Reg_outputs2,
										  Reg_outputs3,
										  Reg_outputs4,
										  Reg_outputs5,
										  Reg_outputs6,
										  Reg_outputs7,
										  Reg_outputs8,
										  Reg_outputs9])
	return model


def Custom_Adv_Reg(In_Shape):
	#Define shared layers
	Shared_C1 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C2 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP1 = MaxPooling2D(pool_size=(2,2))
	Shared_DP1 = Dropout(0.2, name='Drop1')
	Shared_C3 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C4 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP2 = MaxPooling2D(pool_size=(2,2))
	Shared_DP2 = Dropout(0.2)
	Shared_C5 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C6 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP3 = MaxPooling2D(pool_size=(2,2))
	Shared_DP3 = Dropout(0.2)
	Shared_Flatten = Flatten()
	Shared_Dense = Dense(512, activation='relu')
	Shared_DP4 = Dropout(0.2)
	
	#Built sequential model
	Normal_inputs = Input(shape=In_Shape, name='Normal_inputs')
	Normal_C1 = Shared_C1(Normal_inputs)
	Normal_C2 = Shared_C2(Normal_C1)
	Normal_MP1 = Shared_MP1(Normal_C2)
	Normal_DP1 = Shared_DP1(Normal_MP1)
	Normal_C3 = Shared_C3(Normal_DP1)
	Normal_C4 = Shared_C4(Normal_C3)
	Normal_MP2 = Shared_MP2(Normal_C4)
	Normal_DP2 = Shared_DP2(Normal_MP2)
	Normal_C5 = Shared_C5(Normal_DP2)
	Normal_C6 = Shared_C6(Normal_C5)
	Normal_MP3 = Shared_MP3(Normal_C6)
	Normal_DP3 = Shared_DP3(Normal_MP3)
	Normal_Flatten = Shared_Flatten(Normal_DP3)
	Normal_Dense = Shared_Dense(Normal_Flatten)
	Normal_DP4 = Shared_DP4(Normal_Dense)

	Adversarial_inputs = Input(shape=In_Shape, name='Adversarial_inputs')
	Adversarial_C1 = Shared_C1(Adversarial_inputs)
	Adversarial_C2 = Shared_C2(Adversarial_C1)
	Adversarial_MP1 = Shared_MP1(Adversarial_C2)
	Adversarial_DP1 = Shared_DP1(Adversarial_MP1)
	Adversarial_C3 = Shared_C3(Adversarial_DP1)
	Adversarial_C4 = Shared_C4(Adversarial_C3)
	Adversarial_MP2 = Shared_MP2(Adversarial_C4)
	Adversarial_DP2 = Shared_DP2(Adversarial_MP2)
	Adversarial_C5 = Shared_C5(Adversarial_DP2)
	Adversarial_C6 = Shared_C6(Adversarial_C5)
	Adversarial_MP3 = Shared_MP3(Adversarial_C6)
	Adversarial_DP3 = Shared_DP3(Adversarial_MP3)
	Adversarial_Flatten = Shared_Flatten(Adversarial_DP3)
	Adversarial_Dense = Shared_Dense(Adversarial_Flatten)
	Adversarial_DP4 = Shared_DP4(Adversarial_Dense)
	
	normal_output = Dense(nb_classes, activation='softmax', name='normal_output')(Normal_DP4)
	regularised_output = Dense(nb_classes, activation='softmax', name='regularised_output')(Normal_DP4)
	adversarial_output = Dense(nb_classes, activation='softmax', name='adversarial_output')(Adversarial_DP4)

	model = Model(inputs=[Normal_inputs,Adversarial_inputs], outputs=[normal_output, regularised_output, adversarial_output])
	return model


def Multi_Adv_Reg(In_Shape):
	#Define shared layers
	Shared_C1 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C2 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP1 = MaxPooling2D(pool_size=(2,2))
	Shared_DP1 = Dropout(0.2, name='Drop1')
	Shared_C3 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C4 = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP2 = MaxPooling2D(pool_size=(2,2))
	Shared_DP2 = Dropout(0.2)
	Shared_C5 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_C6 = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
	Shared_MP3 = MaxPooling2D(pool_size=(2,2))
	Shared_DP3 = Dropout(0.2)
	Shared_Flatten = Flatten()
	Shared_Dense = Dense(512, activation='relu')
	Shared_DP4 = Dropout(0.2)
	Shared_output = Dense(nb_classes)
	Shared_regularise_output = Dense(nb_classes)
	Logits_transformer1 = Dense(100, activation='sigmoid')
	Logits_transformer2 = Dense(nb_classes)
	Shared_multiply = Multiply()

	#Built sequential model
	Normal_inputs = Input(shape=In_Shape, name='Normal_inputs')
	Normal_C1 = Shared_C1(Normal_inputs)
	Normal_C2 = Shared_C2(Normal_C1)
	Normal_MP1 = Shared_MP1(Normal_C2)
	Normal_DP1 = Shared_DP1(Normal_MP1)
	Normal_C3 = Shared_C3(Normal_DP1)
	Normal_C4 = Shared_C4(Normal_C3)
	Normal_MP2 = Shared_MP2(Normal_C4)
	Normal_DP2 = Shared_DP2(Normal_MP2)
	Normal_C5 = Shared_C5(Normal_DP2)
	Normal_C6 = Shared_C6(Normal_C5)
	Normal_MP3 = Shared_MP3(Normal_C6)
	Normal_DP3 = Shared_DP3(Normal_MP3)
	Normal_Flatten = Shared_Flatten(Normal_DP3)
	Normal_Dense = Shared_Dense(Normal_Flatten)
	Normal_DP4 = Shared_DP4(Normal_Dense)

	normal_logits = Shared_output(Normal_DP4)
	regularised_logits = Shared_regularise_output(Normal_DP4)
	regularised_output = Activation('softmax', name='regularised_output')(regularised_logits)
	
	Logits_Trans1 = Logits_transformer1(regularised_logits)
	Logits_Trans2 = Logits_transformer2(Logits_Trans1)
	Multi_logits = Shared_multiply([normal_logits, Logits_Trans2])
	normal_output = Activation('softmax', name='normal_output')(Multi_logits)

	Adversarial_inputs = Input(shape=In_Shape, name='Adversarial_inputs')
	Adversarial_C1 = Shared_C1(Adversarial_inputs)
	Adversarial_C2 = Shared_C2(Adversarial_C1)
	Adversarial_MP1 = Shared_MP1(Adversarial_C2)
	Adversarial_DP1 = Shared_DP1(Adversarial_MP1)
	Adversarial_C3 = Shared_C3(Adversarial_DP1)
	Adversarial_C4 = Shared_C4(Adversarial_C3)
	Adversarial_MP2 = Shared_MP2(Adversarial_C4)
	Adversarial_DP2 = Shared_DP2(Adversarial_MP2)
	Adversarial_C5 = Shared_C5(Adversarial_DP2)
	Adversarial_C6 = Shared_C6(Adversarial_C5)
	Adversarial_MP3 = Shared_MP3(Adversarial_C6)
	Adversarial_DP3 = Shared_DP3(Adversarial_MP3)
	Adversarial_Flatten = Shared_Flatten(Adversarial_DP3)
	Adversarial_Dense = Shared_Dense(Adversarial_Flatten)
	Adversarial_DP4 = Shared_DP4(Adversarial_Dense)
	
	adversarial_logits = Shared_output(Adversarial_DP4)
	adv_regularised_logits = Shared_regularise_output(Adversarial_DP4)
	adv_regularised_output = Activation('softmax', name='adv_regularised_output')(adv_regularised_logits)
	adv_Logits_Trans1 = Logits_transformer1(adv_regularised_logits)
	adv_Logits_Trans2 = Logits_transformer2(adv_Logits_Trans1)
	Adv_Multi_logits = Shared_multiply([adversarial_logits, adv_Logits_Trans2])
	adversarial_output = Activation('softmax', name='adversarial_output')(Adv_Multi_logits)

	model = Model(inputs=[Normal_inputs,Adversarial_inputs], outputs=[normal_output, regularised_output, adversarial_output, adv_regularised_output])
	return model


def Multi_Adv_Reg2(In_Shape):
	def AE_Detector(inputs):
		Logits_transformer1 = Dense(100, activation='sigmoid')(inputs)
		outputs = Dense(nb_classes)(Logits_transformer1)
		return outputs

	#Define shared layers
	inputs = Input(shape=In_Shape)
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
	x = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = Dropout(0.2, name='Drop1')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = Dropout(0.2)(x)
	x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	outputs = Dropout(0.2)(x)
	shared_hiddenlayers =  Model(inputs, outputs)

	#Built sequential model
	Normal_inputs = Input(shape=In_Shape, name='Normal_inputs')
	Normal_hidden = shared_hiddenlayers(Normal_inputs)
	Adversarial_inputs = Input(shape=In_Shape, name='Adversarial_inputs')
	Adversarial_hidden = shared_hiddenlayers(Adversarial_inputs)
	
	adversarial_logits = Dense(nb_classes)(Adversarial_hidden)
	adversarial_output = Activation('softmax', name='adversarial_output')(adversarial_logits)

	normal_logits = Dense(nb_classes)(Normal_hidden)
	regularised_logits = Dense(nb_classes)(Normal_hidden)
	regularised_output = Activation('softmax', name='regularised_output')(regularised_logits)
	
	Logits_Trans2 = AE_Detector(regularised_logits)

	Multi_logits = Multiply()([normal_logits, Logits_Trans2])
	normal_output = Activation('softmax', name='normal_output')(Multi_logits)

	model = Model(inputs=[Normal_inputs,Adversarial_inputs], outputs=[normal_output, regularised_output, adversarial_output])
	return model
## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.
import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time

from keras import backend as K
from setup_cifar import CIFAR, CIFARModel, CIFARModel_Oracle, CIFAR2
from setup_mnist import MNIST, MNISTModel, MNISTModel_Oracle, MNIST2
from setup_inception import ImageNet, InceptionModel
from keras.models import load_model

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    X_benign = data.test_data[start:samples]

    return X_benign, inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        K.set_learning_phase(1)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        #data, model =  MNIST(), MNISTModel_Oracle("models/mnist_oracle", sess)
        
        data = MNIST()
        #model = MNISTModel("models/mnist", sess)
        model = MNISTModel_Oracle("models/mnist_oracle", sess)
        model.model = load_model('models/Colabel adversarially trained model (mnist)(Single input output).h5')

#'''
        attack = CarliniL2(sess, model, batch_size=12, max_iterations=1000, confidence=5, targeted=True)

        X_test, inputs, targets = generate_data(data, samples=12, targeted=True,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                start=0, inception=False)

        print(X_test.shape, inputs.shape, targets.shape)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        y_adv = model.model.predict(adv)
        z_adv = np.argmax(y_adv, axis=1)
        z_target = np.argmax(targets, axis=1)

        fig = plt.figure(figsize=(10, 2))
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.05, hspace=0.05)

        rowsize = 10
        start =0
        for j in range(0,10):
            for i in range(0,10):
                ax = fig.add_subplot(gs[j, i])
                m = adv[rowsize*j+i].reshape((28,28))
                ax.imshow(m, interpolation='none', cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()
    #ax.set_xlabel('{0}'.format(Target_labels_o[i]), fontsize=16)

        #print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        #print(adv.shape)
        #np.savez('appendix/Defended MNIST non-targeted 40k 100 images.npz',
        #         X_test=X_test, X_adv=adv, labels=z_target)
        

'''
        Adv_data = np.load('black_box_data/MNIST_CW-0k.npz')
        (X_test, X_adv, y_true) = (Adv_data['X_test'],
                                   Adv_data['X_adv'],
                                   Adv_data['y_true'])

        print(X_adv.shape)
        m = data.test_data[1].reshape((28,28))
        plt.imshow(m, cmap='gray')
        plt.show()
        ma = X_adv[1].reshape((28,28))
        plt.imshow(ma, cmap='gray')
        plt.show()

        #print(model.model.evaluate(data.test_data, data.test_labels))
        #print(model.model.evaluate(X_adv, y_true))
'''
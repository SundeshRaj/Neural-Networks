# Raj, Sundesh
# 1001-633-297
# 2019-12_02
# Assignment-04-03

import pytest
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import SGD
from cnn import CNN
import os

def test_train():
    my_cnn = CNN()
    classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0-0.5
    X_test /= 255.0-0.5
    number_of_train_samples_to_use = 500
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu',input_shape=(32, 32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    train = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test,y_test))
    
    my_cnn.add_input_layer(shape=X_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size = 3, padding='same', strides=1, activation='relu', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='pool1')
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size = 3, padding='same', strides=1, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name='flat')
    my_cnn.append_dense_layer(num_nodes=512, activation='relu', trainable=True, name='dense1')
    my_cnn.append_dense_layer(num_nodes=classes, activation='softmax', trainable=True, name='dense2')
    my_cnn.model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    train1 = my_cnn.model.fit(X_train, y_train, batch_size=32,epochs=50)
#    np.testing.assert_almost_equal(train1.history['loss'], train.history['loss'], decimal = 1)
    assert np.allclose(train.history['loss'], train1.history['loss'], rtol=1e-1, atol=1e-1*6)
    
def test_evaluate():
    my_cnn = CNN()
    classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255-0.5
    X_test /= 255-0.5
    number_of_train_samples_to_use = 500
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu',input_shape=(32, 32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=32, epochs=50)
    evaluate = model.evaluate(X_test, y_test) 
    
    my_cnn.add_input_layer(shape=X_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size = 3, padding='same', strides=1, activation='relu', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='pool1')
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size = 3, padding='same', strides=1, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name='flat')
    my_cnn.append_dense_layer(num_nodes=512, activation='relu', trainable=True, name='dense1')
    my_cnn.append_dense_layer(num_nodes=classes, activation='softmax', trainable=True, name='dense2')
    my_opt = SGD(lr=0.01, momentum=0.0)
    my_cnn.model.compile(optimizer = my_opt, loss='categorical_crossentropy', metrics = ['accuracy'])
    my_cnn.model.fit(X_train, y_train,batch_size=32,epochs=50)
    my_eval = my_cnn.model.evaluate(X_test, y_test)
    assert np.allclose(evaluate, my_eval, rtol=1e-1, atol=1e-1*6)
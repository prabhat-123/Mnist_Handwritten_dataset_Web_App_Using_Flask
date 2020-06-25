import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import mnist
import sys

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto(gpu_options = 
#                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# # device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


def one_hot(data, num_categories):
    oh = np.zeros((len(data),num_categories))
    for i,entry in enumerate(data):
        oh[i, entry] = 1
    return oh


# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# preprocess data
x_train = x_train.reshape( (60000,28,28,1) ) / 256
x_test = x_test.reshape( (10000,28,28,1) ) / 256
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# build the model
model = Sequential()
input_shape=(28,28,1)
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),
                       strides=(2,2)))
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                       strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,
                activation='softmax'))

# load model weight

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# train
num_epochs = 1
if num_epochs != 0:
    # train the model
    model.fit(x_train, y_train,
              batch_size=4,
              epochs=num_epochs)

# evaluate model
score = model.evaluate(x_test, y_test)
print('\nScore: ', score)
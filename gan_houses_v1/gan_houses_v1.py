import os
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer, Dropout, BatchNormalization
from keras.regularizers import L1L2
from keras.constraints import non_neg
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

# load training data
try:
    with open('x_train', 'rb') as x_trainFile:
        x_train = pickle.load(x_trainFile)
    with open('x_validation', 'rb') as x_validationFile:
        x_validation = pickle.load(x_validationFile)

    # this is the normalization vector, all the numerical data of each house has been normalized
    # to reduce error, to get a normal house at the end of training, simply take your predicted house
    # and multiply its numeric attributes by this vector, the data is already aligned
    with open('x_normalization', "rb") as normalizationVectorFile:
        normalizationVector = pickle.load(normalizationVectorFile)
except FileNotFoundError:
    print("File not found ! Please rerun and add the extractData() method")
    exit(-1)

# defining the generator
generator = Sequential()
generator.add(Dense(20, input_dim=25, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5), ))
generator.add(BatchNormalization())
generator.add(Dense(25, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Dense(30, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Dense(35, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Dense(40, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Reshape((40,)))

# defining the discriminator
discriminator = Sequential()
discriminator.add(InputLayer(input_shape=(40,)))
generator.add(BatchNormalization())
discriminator.add(Dense(40, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(35, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(30, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(25, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(20, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5), bias_initializer='ones', bias_constraint=non_neg()))

gan = simple_gan(generator, discriminator, normal_latent_sampling((25,)))
model = AdversarialModel(base_model=gan, player_params=[generator.trainable_weights, discriminator.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
test = model.fit(x=x_train, y=gan_targets(np.array(x_train).shape[0]), epochs=1000, batch_size=50, shuffle=True)

discriminator.save('discriminator.h5')
generator.save('generator.h5')

plt.plot(test.history['player_0_loss'])
plt.plot(test.history['player_1_loss'])
plt.plot(test.history['loss'])
plt.show()

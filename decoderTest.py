import House
import pickle
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, GaussianNoise, InputLayer

decoder = Sequential()
decoder.add(InputLayer((20,)))
decoder.add(GaussianNoise(0))
decoder = load_model('decoder')

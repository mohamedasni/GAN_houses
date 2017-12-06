import House
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense

def extractData():
    houses = []
    with open("train.csv", 'r') as data:
        for line in data:
            houses.append(House.House(line))
    numericalData = []
    categoricalData = []
    # first, we get all the categorical data from the houses because we need to transform them into one hot encoding
    # we will get the numerical values letter
    # and we need to get all
    for house in houses:
        categoricalData.append(house.categoricalData())
        numericalData.append(house.numberData())

    # go through all the data and determine how many classes there are for all the dimensions for each dimension
    dimensionList = []
    for dimension in range(len(categoricalData[0])):
        tempFullColumn = []
        for house in houses:
            if house.categoricalData()[dimension] not in tempFullColumn:
                tempFullColumn.append(house.categoricalData()[dimension])
        dimensionList.append(tempFullColumn)  # this dictionary all available answers for each dimensions
                               # this will be useful to get the number of classes for the one hot encoding

    # we have to normalize the input to prevent the number from getting big and increase our error for no reason
    # here we add together all the square of each value
    squareSum = []
    for i in range(len(numericalData[0])):
        squareSum.append(0)
    for dimension in numericalData:
        index = 0
        for value in dimension:
            squareSum[index] += value**2
            index += 1
    # apply the square the results
    for i in squareSum:
        squareSum[squareSum.index(i)] = i**0.5

    # now, we prepare the houses to get fed into the autoencoder
    x_data = []
    for house in houses:
        singleHouseData = house.numberData()
        singleHouseData.append(house.categoricalData())
        for i in squareSum:
            singleHouseData[squareSum.index(i)] = singleHouseData[squareSum.index(i)] / i
        counter = 0
        for dimension in dimensionList:
            singleHouseData.append(dimension.index(house.categoricalData()[counter]))
            counter += 1
        x_data.append(singleHouseData)

    x_train = x_data[:1300]
    x_validation = x_data[1300:]
    print(len(x_train))


    # save the data so we dont have to parse it again.
    with open('x_train', "wb") as pickle_out:
        pickle.dump(x_train, pickle_out)

    # save the data so we dont have to parse it again.
    with open('x_validation', "wb") as pickle_out:
        pickle.dump(x_validation, pickle_out)

    with open('x_normalization', "wb") as pickle_out:
        pickle.dump(squareSum, pickle_out)

#TODO uncomment to if you got FileNotFound error or other type of errors
extractData()
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


encoding_dim = 20  # tries to compress the house data to 20 dimension instead of 39

input_size = len(x_train[0]) # the number of dimensions of the data
input_data = Input(shape=(input_size,)) # placeholder for the autoencoder input

print(input_size)

# create a model for the encoder part of the
encoder = Sequential()
encoder.add(Dense(input_size, activation='relu', input_shape=(input_size,), kernel_initializer='random_uniform'))
encoder.add(Dense(35, activation='relu', kernel_initializer='random_uniform'))
encoder.add(Dense(30, activation='relu', kernel_initializer='random_uniform'))
encoder.add(Dense(25, activation='relu', kernel_initializer='random_uniform'))
encoder.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))


decoder = Sequential()
decoder.add(Dense(20, activation='relu', input_shape=(20,), kernel_initializer='random_uniform'))
decoder.add(Dense(25, activation='relu', kernel_initializer='random_uniform'))
decoder.add(Dense(30, activation='relu', kernel_initializer='random_uniform'))
decoder.add(Dense(35, activation='relu', kernel_initializer='random_uniform'))
decoder.add(Dense(input_size, activation='relu', kernel_initializer='random_uniform'))

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

# now lets compile the autoencoder
autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
decoder.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=100,
#                 shuffle=True,
#                 validation_data=(x_validation, x_validation))
#
# decoder.save("decoder")

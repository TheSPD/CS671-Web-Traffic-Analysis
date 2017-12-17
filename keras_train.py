from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from preProcessing import PreProcessor
from dataGenerator import DataGenerator

# Initializing the Pre Processor
preProc = PreProcessor(num_articles=50)

# Save Articles to data folder
preProc.saveArticles()

# Initialising the RNN
regressor = Sequential()

seq_length = 7
data_dim = 15

# Adding the input layerand the LSTM layer
regressor.add(LSTM(units=32,
                   activation='relu',
                   input_shape=(seq_length, data_dim)))


# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

num_articles = 50
num_ts = 543
list_IDs = range(num_ts * num_articles)
batch_size = 32

#
training_generator = DataGenerator(batch_size=batch_size).generate(list_IDs)

# Fitting the RNN to the Training set
regressor.fit_generator(generator=training_generator,
                        steps_per_epoch=len(list_IDs) // batch_size,
                        epochs=10)

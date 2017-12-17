import numpy as np


class DataGenerator(object):

    '''Generates data for Keras'''
    def __init__(self, dim_x=7, dim_y=15, batch_size=32, shuffle=True):
        '''Initialization'''
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ts_per_file = 543

    def generate(self, list_IDs):
        '''Generates batches of samples'''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        '''Generates order of exploration'''
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, list_IDs_temp):
        '''Generates data of batch_size samples'''
        # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            file_ID = i // self.ts_per_file
            file = np.load('./data/article_' + str(file_ID) + '.npy')
            ts_ID = i % self.ts_per_file
            X[i, :, :, 0] = np.rot90(file[:, ts_ID : ts_ID + self.dim_x], 3, (0, 1))
            y[i] = file[0, ts_ID + self.dim_x]

        return X, y

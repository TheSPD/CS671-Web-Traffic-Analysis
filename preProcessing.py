# Importing Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import re


class PreProcessor(object):

    num_articles = 0
    num_ts = 0
    seq_length = 0

    def __init__(self, filepath='./data/train_1.csv', seq_length=7,
                 num_articles=0):
        self.filepath = filepath
        self.label_encoder = preprocessing.LabelEncoder()
        self.one_hot_enc = preprocessing.OneHotEncoder()
        self.seq_length = seq_length
        self.num_articles = num_articles

    def _readFile(self):
        ''' Imports dataset '''
        print 'Importing dataset...'
        self.train = pd.read_csv(self.filepath).fillna(0)
        self.page = self.train['Page']
        if(self.num_articles == 0):
            self.num_articles = self.train.shape[0]
        self.num_ts = self.train.shape[1] - self.seq_length

    def _preProcess(self):
        '''Extracting source, access and agent data from Page Information'''

        def __get_source(page):
            res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
            if res:
                return res.group().split('_')[1]
            return 'na'

        def __get_access(page):
            res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
            if res:
                return res.group().split('_')[2]
            return 'na'

        def __get_agent(page):
            res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
            if res:
                return res.group().split('_')[3]
            return 'na'

        print 'Preprocessing...'
        self.source = self.train.Page.map(__get_source)
        self.access = self.train.Page.map(__get_access)
        self.agent = self.train.Page.map(__get_agent)

        self.source = self.label_encoder.fit_transform(self.source)
        self.access = self.label_encoder.fit_transform(self.access)
        self.agent = self.label_encoder.fit_transform(self.agent)

        self.train = self.train.drop('Page', axis=1)

        self.source = np.reshape(self.source, (-1, 1))
        self.access = np.reshape(self.access, (-1, 1))
        self.agent = np.reshape(self.agent, (-1, 1))

        self.source = self.one_hot_enc.fit_transform(self.source)
        self.access = self.one_hot_enc.fit_transform(self.access)
        self.agent = self.one_hot_enc.fit_transform(self.agent)

    def saveArticles(self):
        '''Save articles in data folder'''
        self._readFile()
        self._preProcess()
        for article in range(self.num_articles):
            row = self.train.iloc[article, :].values
            row_len = len(row)
            row = np.reshape(row, (1, row_len))
            sourceCol = np.rot90(np.repeat(self.source[article, :].toarray(),
                                           row_len, axis=0), 1, (0, 1))
            accessCol = np.rot90(np.repeat(self.access[article, :].toarray(),
                                           row_len, axis=0), 1, (0, 1))
            agentCol = np.rot90(np.repeat(self.agent[article, :].toarray(),
                                          row_len, axis=0), 1, (0, 1))
            row = np.append(row, sourceCol, axis=0)
            row = np.append(row, accessCol, axis=0)
            row = np.append(row, agentCol, axis=0)
            print 'Saving article ', article, ' with shape ', row.shape
            np.save('./data/article_' + str(article) + '.npy', row)

    def _saveTimeSteps(self):
        '''Save time steps in data folder
           (Use with care - creates a lot of files)'''
        self._readFile()
        self._preProcess()
        self.saveArticles()
        ts_num = 0
        for article in range(self.num_articles):
            curArticle = np.load('./data/article_' + str(article) + '.npy')
            print curArticle.shape
            print 'Creating timeseries for article ', article, '...'
            for i in range(0, curArticle.shape[1] - self.seq_length):
                ts_data = curArticle[:, i: i + self.seq_length]
                ts_data = np.rot90(ts_data, 3, (0, 1))
                ts_output = curArticle[0, i + self.seq_length]
                np.save('./data/ts_data_' + str(ts_num) + '.npy',
                        ts_data)
                np.save('./data/ts_output_' + str(ts_num) + '.npy',
                        ts_output)
                ts_num += 1

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import re

# Importing dataset
print 'Importing dataset...'
train = pd.read_csv('./data/train_1.csv').fillna(0)
page = train['Page']


# Extracting source, access and agent data from Page Information


def get_source(page):
    res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
    if res:
        return res.group().split('_')[1]
    return 'na'


def get_access(page):
    res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
    if res:
        return res.group().split('_')[2]
    return 'na'


def get_agent(page):
    res = re.search('_[a-z]+\.[a-z]+\.[a-z]+_[a-z\-]+_[a-z]+', page)
    if res:
        return res.group().split('_')[3]
    return 'na'


print 'Preprocessing...'
source = train.Page.map(get_source)
access = train.Page.map(get_access)
agent = train.Page.map(get_agent)

le = preprocessing.LabelEncoder()
source = le.fit_transform(source)
access = le.fit_transform(access)
agent = le.fit_transform(agent)

train = train.drop('Page', axis=1)

source = np.reshape(source, (-1, 1))
access = np.reshape(access, (-1, 1))
agent = np.reshape(agent, (-1, 1))

enc = preprocessing.OneHotEncoder()

source = enc.fit_transform(source)
access = enc.fit_transform(access)
agent = enc.fit_transform(agent)

# for article in range(len(train)):
for article in range(50):
    row = train.iloc[article, :].values
    row_len = len(row)
    row = np.reshape(row, (1, row_len))
    sourceCol = np.rot90(np.repeat(source[article, :].toarray(),
                                   row_len, axis=0), 1, (0, 1))
    accessCol = np.rot90(np.repeat(access[article, :].toarray(),
                                   row_len, axis=0), 1, (0, 1))
    agentCol = np.rot90(np.repeat(agent[article, :].toarray(),
                                  row_len, axis=0), 1, (0, 1))
    row = np.append(row, sourceCol, axis=0)
    row = np.append(row, accessCol, axis=0)
    row = np.append(row, agentCol, axis=0)
    print 'Saving article ', article, ' with shape ', row.shape
    np.save('./data/article_' + str(article) + '.npy', row)

seq_length = 7
ts_num = 0

# for article in range(len(train)):
# for article in range(50):
#     curArticle = np.load('./data/article_' + str(article) + '.npy')
#     print curArticle.shape
#     print 'Creating timeseries for article ', article, '...'
#     for i in range(0, curArticle.shape[1] - seq_length):
#         ts_data = curArticle[:, i: i + seq_length]
#         ts_data = np.rot90(ts_data, 3, (0, 1))
#         ts_output = curArticle[0, i + seq_length]
#         np.save('./data/ts_data_' + str(ts_num) + '.npy',
#                 ts_data)
#         np.save('./data/ts_output_' + str(ts_num) + '.npy',
#                 ts_output)
#         ts_num += 1
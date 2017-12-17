from preProcessing import PreProcessor

pp = PreProcessor(num_articles=50)

pp.saveArticles()

print pp.num_articles, pp.num_ts, pp.seq_length, pp.data_dim


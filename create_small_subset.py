import pandas as pd
import numpy as np

# the file is from https://www.kaggle.com/kazanova/sentiment140
df = pd.read_csv('training.1600000.processed.noemoticon.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'])
remove_punc(tweets)

is_negative = df['target'] == 0
is_positive = df['target'] == 4

negatives = df[is_negative]
positives = df[is_positive]


small_positives = positives[:5000]
small_negatives = negatives[:5000]

# create a shuffled sample containing the same proportion of positive and negative than in the original file
small_example = pd.concat([small_positives, small_negatives])
small_example = small_example.sample(frac=1)

small_example.to_csv(path_or_buf='training_set_small.csv', header=False)

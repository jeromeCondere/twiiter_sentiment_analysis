import pandas as pd
import numpy as np

# the file is from https://www.kaggle.com/kazanova/sentiment140
df = pd.read_csv('training.1600000.processed.noemoticon.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'])

is_negative_test = df['target'] == 0
is_positive_test = df['target'] == 4

negatives = df[is_negative_test]
positives = df[is_positive_test]


test_positives = positives[5001:30000]
test_negatives = negatives[5001:30000]

# create a shuffled sample containing the same proportion of positive and negative than in the original file
test_example = pd.concat([test_positives, test_negatives])
test_example = test_example.sample(frac=1)

test_example.to_csv(path_or_buf='training_set_test.csv', header=False)
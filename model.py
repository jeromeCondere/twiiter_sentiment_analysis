import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from langdetect import detect
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, GlobalMaxPool1D, Conv1D, LSTM, SpatialDropout1D
from keras.layers import Dropout, Bidirectional, BatchNormalization
import nltk
import emoji


plt.style.use('ggplot')


def cleaner(tweet):
   tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
   tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
   tweet = " ".join(tweet.split())
   tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis
   tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
   tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet))
   return tweet



def create_embedding_matrix(filepath, word_index, embedding_dim):
   vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
   embedding_matrix = np.zeros((vocab_size, embedding_dim))

   with open(filepath) as f:
      for line in f:
         word, *vector = line.split()
         if word in word_index:
            idx = word_index[word] 
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
   
   return embedding_matrix


def plot_history(history):
   acc = history.history['accuracy']
   val_acc = history.history['val_accuracy']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   x = range(1, len(acc) + 1)

   plt.figure(figsize=(12, 5))
   plt.subplot(1, 2, 1)
   plt.plot(x, acc, 'b', label='Training acc')
   plt.plot(x, val_acc, 'r', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.legend()
   plt.subplot(1, 2, 2)
   plt.plot(x, loss, 'b', label='Training loss')
   plt.plot(x, val_loss, 'r', label='Validation loss')
   plt.title('Training and validation loss')
   plt.legend()


def transform_sentence_to_sequence(sentence, tokenizer, maxlen):
   texts = tokenizer.texts_to_sequences([sentence])
   texts = pad_sequences(texts, padding='post', maxlen=maxlen)
   return texts


# 0 = negative, 2 = neutral, 4 = positive

tweets = pd.read_csv('training_set_small.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'])

tweets['target'].replace({4: 1}, inplace=True)
tweets['text'] = tweets['text'].map(lambda x: cleaner(x))
text = tweets['text']
target = tweets['target']

tweets_train, tweets_test, target_train, target_test = train_test_split(text, target, test_size=0.25, random_state=1000)


maxlen = 60
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_train)
vocab_size = len(tokenizer.word_index) + 1

# transform tweets to vector
X_train = tokenizer.texts_to_sequences(tweets_train)
X_test = tokenizer.texts_to_sequences(tweets_test)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embedding_dim = 100

model = Sequential()
# convert sequence of integer into sequence of dense vector of size embedding_dim
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# using binary crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, target_train, epochs=10, validation_data=(X_test, target_test), batch_size=100)
loss, accuracy = model.evaluate(X_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, target_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)




# second model with global max pooling
model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model2.add(GlobalMaxPool1D())
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history2 = model2.fit(X_train, target_train, epochs=10, validation_data=(X_test, target_test), batch_size=100)
loss2, accuracy2 = model2.evaluate(X_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy2))
loss2, accuracy2 = model2.evaluate(X_test, target_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
plot_history(history2)




embedding_dim_glove = 50
embedding_matrix = create_embedding_matrix('glove.6B.50d.txt', tokenizer.word_index, embedding_dim_glove)

model3 = Sequential()
model3.add(Embedding(vocab_size, embedding_dim_glove, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model3.add(GlobalMaxPool1D())
model3.add(Dense(10, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model3.summary()


history3 = model3.fit(X_train, target_train, epochs=25, validation_data=(X_test, target_test), batch_size=150)
loss3, accuracy3 = model3.evaluate(X_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy3))
loss3, accuracy3 = model3.evaluate(X_test, target_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy3))
plot_history(history3)


model4 = Sequential()
model4.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model4.add(Conv1D(128, 5, activation='relu'))
model4.add(GlobalMaxPool1D())
model4.add(Dense(10, activation='relu'))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model4.summary()


history4 = model4.fit(X_train, target_train, epochs=15, validation_data=(X_test, target_test), batch_size=150)
loss4, accuracy4 = model4.evaluate(X_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy4))
loss4, accuracy4 = model4.evaluate(X_test, target_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy4))
plot_history(history4)




model5 = Sequential()
model5.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model5.add(SpatialDropout1D(0.1))
model5.add(Conv1D(128, 5, activation='relu'))
model5.add(Bidirectional(LSTM(4, return_sequences=True)))
model5.add(Conv1D(128, 5, activation='relu'))
model5.add(GlobalMaxPool1D())
model5.add(Dropout(0.25))
model5.add(BatchNormalization())
model5.add(Dense(10, activation='relu'))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model5.summary()


history5 = model5.fit(X_train, target_train, epochs=15, validation_data=(X_test, target_test), batch_size=150)


loss5, accuracy5 = model5.evaluate(X_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy5))
loss5, accuracy5 = model5.evaluate(X_test, target_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy5))
plot_history(history5)




to_pred = transform_sentence_to_sequence( "With joy, it's good weather", tokenizer, 60)
model4.predict(to_pred)
model5.predict(to_pred)

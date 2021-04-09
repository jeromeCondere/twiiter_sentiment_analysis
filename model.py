import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, GlobalMaxPool1D, Conv1D, LSTM, SpatialDropout1D
from keras.layers import Dropout, Bidirectional, BatchNormalization
from utils import *
from keras import Model


maxlen = 60
embedding_dim = 100
embedding_dim_glove = 50

def get_first_model(vocab_size: int, embedding_dim: int, maxlen: int) -> Model:
   model = Sequential()
   model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
   model.add(GlobalMaxPool1D())
   model.add(Dense(10, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

def get_second_model(vocab_size: int, embedding_dim: int, maxlen: int, embedding_matrix) -> Model:
   model = Sequential()
   model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
   model.add(Conv1D(128, 5, activation='relu'))
   model.add(GlobalMaxPool1D())
   model.add(Dense(10, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

def get_third_model(vocab_size: int, embedding_dim: int, maxlen: int, embedding_matrix) -> Model:
   model = Sequential()
   model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
   model.add(SpatialDropout1D(0.1))
   model.add(Conv1D(128, 5, activation='relu'))
   model.add(Bidirectional(LSTM(4, return_sequences=True)))
   model.add(Conv1D(128, 5, activation='relu'))
   model.add(GlobalMaxPool1D())
   model.add(Dropout(0.25))
   model.add(Dense(10, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

def main():
   tweets = pd.read_csv('training_set_small.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'])

   tweets['target'].replace({4: 1}, inplace=True)
   tweets['text'] = tweets['text'].map(lambda x: cleaner(x))
   text = tweets['text']
   target = tweets['target']
   
   tweets_train, tweets_test, target_train, target_test = train_test_split(text, target, test_size=0.25, random_state=1000)
   
   
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(tweets_train)
   vocab_size = len(tokenizer.word_index) + 1
   
   # transform tweets to vector
   X_train = tokenizer.texts_to_sequences(tweets_train)
   X_test = tokenizer.texts_to_sequences(tweets_test)
   X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
   X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
   embedding_matrix = create_embedding_matrix('glove.6B.50d.txt', tokenizer.word_index, embedding_dim_glove)
   
   model = get_first_model(vocab_size, embedding_dim, maxlen)
   model2 = get_second_model(vocab_size, embedding_dim_glove, maxlen, embedding_matrix)
   model3 = get_third_model(vocab_size, embedding_dim_glove, maxlen, embedding_matrix)
   
   history = model.fit(X_train, target_train, epochs=10, verbose=False, validation_data=(X_test, target_test), batch_size=150)
   history2 = model2.fit(X_train, target_train, epochs=10, verbose=False, validation_data=(X_test, target_test), batch_size=150)
   history3 = model3.fit(X_train, target_train, epochs=15, verbose=False, validation_data=(X_test, target_test), batch_size=150)
   
   model.save('model1')
   model2.save('model2')
   model3.save('model3')
 



if __name__ == "__main__":
    main()


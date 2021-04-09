import re
import nltk
import emoji
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

def predict_all_model(texts, list_model, tokenizer):
   list_pred = []
   for model in list_model:
      texts_to_pred = transform_sentences_to_sequences(texts, tokenizer, maxlen=maxlen)
      pred = model.predict(texts_to_pred)
      pred = pred.reshape(-1)
      pred[pred > 0.5] = 1
      pred[pred <= 0.5] = 0
      list_pred.append(pred)
   return list_pred


def get_confusion_matrix(list_prediction, actual_value):
   list_confusion_matrix = []
   for index, prediction in enumerate(list_prediction, start=1):
      column_name = 'Predicted_model{}'.format(index)
      data = {'y_Actual': actual_value.to_numpy(), 'y_Predicted_model': prediction}
      df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted_model'])
      confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted_model'], rownames=['Actual'], colnames=[column_name])
      list_confusion_matrix.append(confusion_matrix)
   return list_confusion_matrix

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

def transform_sentences_to_sequences(sentence, tokenizer, maxlen):
   texts = tokenizer.texts_to_sequences(sentence)
   texts = pad_sequences(texts, padding='post', maxlen=maxlen)
   return texts


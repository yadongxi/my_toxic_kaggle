
import sys, os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Embedding, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.layers import Input, concatenate, GlobalMaxPool1D, SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 500
embed_dim = 50
vocab_size = 100000
train = pd.read_csv("input/clean_train.csv")
test = pd.read_csv("input/clean_test.csv")
test["id"] = test["id"].values.astype(str)
EMBEDDING_FILE= 'input/glove.6B.50d.txt'

train_test = pd.concat([train.iloc[:,1], test.iloc[:, 1]], axis=0).astype(str).values

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_test)

train_seq = tokenizer.texts_to_sequences(train.iloc[:,1].astype(str))
test_seq = tokenizer.texts_to_sequences(test.iloc[:, 1].astype(str))

X_train = pad_sequences(train_seq, maxlen=max_len)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_train = train[list_classes].values
X_test = pad_sequences(test_seq, maxlen=max_len)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_dim))
for word, i in word_index.items():
    if i >= vocab_size: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    

x_train = X_train
y_train = Y_train
inputs = Input(shape=(max_len,))
embed = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], trainable=True)(inputs)
tmp = []
for width in [3,4,5]:
    rep = Conv1D(128, width, activation='relu')(embed)
    rep = MaxPooling1D(width)(rep)
    rep = Dropout(0.5)(rep)
    rep = Conv1D(128, width, activation='relu')(rep)
    rep = MaxPooling1D(width)(rep)
    rep = Dropout(0.5)(rep)
    rep = Conv1D(128, width, activation='relu')(rep)
    rep = MaxPooling1D(width)(rep)
    rep = Dropout(0.5)(rep)
    rep = Flatten()(rep)
    tmp.append(rep)
rep = concatenate(tmp)
rep = Dense(64, activation="relu")(rep)
rep = Dropout(0.5)(rep)
outputs = Dense(6, activation='sigmoid')(rep)
model = Model(inputs=inputs, outputs=outputs)
# compile the model
Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint('model/toxic.hdf', save_best_only=True, 
                           monitor='val_loss', mode='min')

history = model.fit(x_train, y_train, validation_split=0.2,batch_size=128,
                    epochs=100, verbose=1,callbacks=[early_stopping,save_best])

#     make a prediction on y (target column)
model.load_weights(filepath = 'model/toxic.hdf')
predictions = model.predict(X_test)
preds = predictions
    
df_results = pd.DataFrame({'id':test.id,
                            'toxic':preds[:,0],
                           'severe_toxic':preds[:,1],
                           'obscene':preds[:,2],
                           'threat':preds[:,3],
                           'insult':preds[:,4],
                           'identity_hate':preds[:,5]}).set_index('id')
# Pandas automatically sorts the columns alphabetically by column name.
# Therefore, we need to re-order the columns to match the sample submission file.
df_results = df_results[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

# create a submission csv file
df_results.to_csv('kaggle_submission.csv', 
                  columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']) 
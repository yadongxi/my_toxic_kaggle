
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Embedding, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 400
embed_dim = 100
vocab_size = 30000
train = pd.read_csv("input/clean_train.csv")
test = pd.read_csv("input/clean_test.csv")
test["id"] = test["id"].values.astype(str)
train_test = pd.concat([train.iloc[:,1], test.iloc[:, 1]], axis=0).astype(str).values

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_test)

train_seq = tokenizer.texts_to_sequences(train.iloc[:,1].astype(str))
test_seq = tokenizer.texts_to_sequences(test.iloc[:, 1].astype(str))

X_train = pad_sequences(train_seq, maxlen=max_len)
Y_train = train.iloc[:, 2:]
X_test = pad_sequences(test_seq, maxlen=max_len)

preds = []
for i in range(6):
    print("step  ", i )
    y = Y_train.iloc[:, i]
    x_train, x_eval, y_train ,y_eval = train_test_split(X_train, y, test_size=0.1, shuffle=True,random_state=5, stratify=y)

    # cnn model
    model = Sequential()
    e = Embedding(vocab_size, embed_dim, input_length=max_len)
    model.add(e)
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    save_best = ModelCheckpoint('model/toxic.hdf', save_best_only=True, 
                               monitor='val_loss', mode='min')

    history = model.fit(x_train, y_train, validation_data=(x_eval, y_eval),batch_size=32,
                        epochs=100, verbose=1,callbacks=[early_stopping,save_best])

    # make a prediction on y (target column)
#    model.load_weights(filepath = 'toxic.hdf')
    predictions = model.predict(X_test)
    y_preds = predictions[:,0]
    # append the prediction to a python list
    preds.append(y_preds)
    
df_results = pd.DataFrame({'id':test.id,
                            'toxic':preds[0],
                           'severe_toxic':preds[1],
                           'obscene':preds[2],
                           'threat':preds[3],
                           'insult':preds[4],
                           'identity_hate':preds[5]}).set_index('id')

# Pandas automatically sorts the columns alphabetically by column name.
# Therefore, we need to re-order the columns to match the sample submission file.
df_results = df_results[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

# create a submission csv file
df_results.to_csv('kaggle_submission.csv', 
                  columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']) 
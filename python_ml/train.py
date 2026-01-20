import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

data_url = "python_ml/data/"

os.makedirs(data_url+"models", exist_ok=True)
df = pd.read_csv(data_url+"raw/IMDB Dataset.csv")

df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['review'])

sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen = 200)
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

model.save(data_url+"models/model.h5")

os.makedirs(data_url+"models/tokenizer_dir", exist_ok=True)

with open(data_url+"models/tokenizer_dir/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer збережено")
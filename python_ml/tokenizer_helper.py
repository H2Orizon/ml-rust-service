import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import json

with open("data/models/tokenizer_dir/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

text = " ".join(sys.argv[1:])

seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, maxlen=200)

print(json.dumps(padded[0].tolist()))
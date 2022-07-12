import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import csv
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from keras.models import model_from_json

pd.options.display.max_colwidth = 200
os.environ["TFHUB_CACHE_DIR"] = "tf_cache"

# Load the text encoder - Universal Senence Encoder by Google -- First time can take as long as 10 minutes
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

print("USE Loaded Successfully")

# Convert csv to pandas dataframe
df = pd.read_csv('labelled_data.csv', delimiter=", ")

print("CSV Read Successfully")

print(df.head())
print(df.shape)

# Positive or Negative?
df['output'] = df['sentiment'].apply (
  lambda x: "negative" if x < 0 else "positive"
)

df = df[['text', 'output']]

# Data preprocessing
type_one_hot = OneHotEncoder(sparse=False).fit_transform (
    df.output.to_numpy().reshape(-1, 1)
)

print("One Hot Successful")

# Split my dataset into 80-20, Training-Testing
train_text, test_text, y_train, y_test = train_test_split (
    df.text,
    type_one_hot,
    test_size=0.2,
    random_state = 42
)

print("Data Successfully Split")

# Encoding my texts
x_test = []

for text in tqdm(test_text):
    emb = use([text])
    text_emb = tf.reshape(emb, [-1]).numpy()
    x_test.append(text_emb)

x_test = np.array(x_test)

print("Universal Sentence Encoding Successful")

# load json and create model
json_file = open('model_num.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_num.h5")
print("Loaded model from disk")

loaded_model.save('model_num.hdf5')
loaded_model=load_model('model_num.hdf5')

predict_sen = loaded_model.predict(x_test[10:20])
classes_x = np.argmax(predict_sen,axis=1)
print(test_text[10:20])
print(classes_x)
print("----")
print("This is x_test")


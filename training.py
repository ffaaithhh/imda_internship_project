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

# Open the csv files
label_csv = open('labelled_data.csv', 'r', encoding='utf-8')
csv_reader = csv.reader(label_csv, delimiter=",")

# Create arrays for X: Text embeddings and Y: Sentiment values
all_embeddings = []
all_sentiments = []

# Load the text encoder - Universal Senence Encoder by Google
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
print("USE has loaded...")

count = 0
# Encode every text and append to array
for row in csv_reader:
    count += 1
    print(f"Count: {count}")

    all_embeddings.append(use([row[1]]))
    all_sentiments.append(row[0])

# Data preprocessing
type_one_hot = OneHotEncoder(sparse=False).fit_transform (
    all_sentiments.to_numpy().reshape(-1, 1)
)

# Split my dataset into 80-20, Training-Testing
train_text, test_text, y_train, y_test = train_test_split (
    all_embeddings,
    type_one_hot,
    test_size=0.2,
    random_state = 42
)

# Encoding my texts
x_train = []
for sen in tqdm(train_text):
    emb = use([sen])
    text_emb = tf.reshape(emb, [-1]).numpy()
    x_train.append(text_emb)

x_train = np.array(x_train)

x_test = []
for sen in tqdm(test_text):
    emb = use([sen])
    text_emb = tf.reshape(emb, [-1]).numpy()
    x_test.append(text_emb)

x_test = np.array(x_test)

print(x_train.shape, x_test.shape)
print(x_train.shape, y_train.shape)

# Model training
model = keras.Sequential()

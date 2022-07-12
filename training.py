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
x_train = []
for sen in tqdm(train_text):
    emb = use([sen])   # error here
    text_emb = tf.reshape(emb, [-1]).numpy()
    x_train.append(text_emb)

x_train = np.array(x_train)

x_test = []
for sen in tqdm(test_text):
    emb = use([sen])
    text_emb = tf.reshape(emb, [-1]).numpy()
    x_test.append(text_emb)

x_test = np.array(x_test)

print("Universal Sentence Encoding Successful")

print(x_train.shape, x_test.shape)
print(x_train.shape, y_train.shape)

# Model training
model = keras.Sequential()

print("Model created!")

model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(x_train.shape[1], ),
    activation='relu'
  )
)

print("Model Step 1")

model.add(
  keras.layers.Dropout(rate=0.5)
)

print("Model Step 2")

model.add(
  keras.layers.Dense(
    units=128,
    activation='relu'
  )
)

print("Model Step 3")

model.add(
  keras.layers.Dropout(rate=0.5)
)

print("Model Step 4")

model.add(keras.layers.Dense(2, activation='softmax'))

print("Model Step 5")

model.compile(
    loss='categorical_crossentropy', 
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

print("Model Step 6")

# error here in model.fit()
history = model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=16, 
    validation_split=0.1, 
    verbose=1, 
    shuffle=True
)

print("Model Step 7")

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Cross-entropy loss")
plt.legend();

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend();

# Testing the model
model.evaluate(x_test, y_test)

# Save the model
model.save('biased_model.h5')
model_json = model.to_json()

with open("model_num.json", "w") as json_file:
  json_file.write(model_json)

model.save_weights("model_num.h5")

print("Completed!")

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import preprocessing

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

class Net(nn.Module):
    def __init__(self, in_count, output_count):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_count, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_count)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        print("Forward!")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)

x_train = Variable(torch.Tensor(x_train).float())
x_test = Variable(torch.Tensor(x_test).float())
y_train = Variable(torch.LongTensor(y_train))
y_test = Variable(torch.LongTensor(y_test))

x = df.text
model = Net(2, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")

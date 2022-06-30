import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow_text

RANDOM_SEED = 12
chunksize = 10 ** 4
chunk_count = 0

print("loading USE...")
# Universal sentence encoder
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

print("USE loaded!...")

with pd.read_csv("labelled_data.csv", chunksize=chunksize, delimiter=", ") as reader:
    print("Reading...")

    # convert the sentiment column from string to decimal
    reader['sentiment'] = reader['sentiment'].astype(float)

    for chunk in reader:
        chunk_count += 1
        print(f"Chunk Count: {chunk_count}")

        # Preprocessing - one hot encoding
        type_one_hot = OneHotEncoder(sparse=False).fit_transform(
            chunk.sentiment.to_numpy().reshape(-1, 1)
        )

        print(f"{chunk_count}...1")

        # Splitting chunk into training and testing
        train_sentiment, test_sentiment, train_text, test_text = train_test_split(
            chunk.sentiment, type_one_hot, test_size=.2, random_state=RANDOM_SEED)

        print(f"{chunk_count}...2")

        x_train = []
        for sen in tqdm(train_sentiment):
            emb = use(sen)
            sentiment_emb = tf.reshape(emb, [-1]).numpy()
            x_train.append(sentiment_emb)

        print(f"{chunk_count}...3")

        x_train = np.array(x_train)

        x_test = []
        for sen in tqdm(test_sentiment):
            emb = use(sen)
            sentiment_emb = tf.reshape(emb, [-1]).numpy()
            x_test.append(sentiment_emb)

        print(f"{chunk_count}...4")

        x_test = np.array(x_test)

        print(x_train.shape, train_sentiment.shape)

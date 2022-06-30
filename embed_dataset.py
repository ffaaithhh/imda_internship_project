import tensorflow_hub as hub
import tensorflow as tf
import csv
import numpy as np

# Import embedding model - Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Open csv
label_csv = open('labelled_data.csv', 'r', encoding='utf-8')
csv_reader = csv.reader(label_csv, delimiter=",")

count = 0
all_embeddings = []
all_sentiments = []

# Encode every text
for row in csv_reader:
    count += 1

    # Check if in bounds
    if len(row) != 2:
        print("Error")
    else:
        sentiment = row[0]
        text = row[1]
        embeded_text = embed([text])
        all_embeddings.append(embeded_text)
        all_sentiments.append(sentiment)

    print(f"Count: {count}")

# Close csv
label_csv.close()

# Save my values
np.save('embeddings.npy', all_embeddings)
np.save('sentiments.npy', all_sentiments)
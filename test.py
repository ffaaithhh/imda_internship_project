import numpy as np

all_embeddings = np.load('embeddings.npy')

count = 0

for emb in all_embeddings:
    count += 1

print("Count: " + str(count))
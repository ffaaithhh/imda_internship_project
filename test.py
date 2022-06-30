import numpy as np

all_embeddings = np.load('embeddings-temp.npy')

for emb in all_embeddings:
    print(emb)
    print("__________")
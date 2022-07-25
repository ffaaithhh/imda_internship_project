# import numpy as np

# all_embeddings = np.load('embeddings-temp.npy')

# for emb in all_embeddings:
#     print(emb)
#     print("__________")

import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

print(embed(["hi"]))
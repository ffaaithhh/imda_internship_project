import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os

def encode_text(text_array):
    encoded_text_array = []
    os.environ["TFHUB_CACHE_DIR"] = "tf_cache"
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    for text in text_array:
        emb = use([text])
        text_emb = tf.reshape(emb, [-1]).numpy()
        encoded_text_array.append(text_emb)
    
    return np.array(encoded_text_array)
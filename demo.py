from keras.models import load_model
from keras.models import model_from_json
import tensorflow_hub as hub
import os
import numpy as np
import csv

# load json and create model
json_file = open('model_num.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into the model
loaded_model.load_weights("model_num.h5")

# Load embedding
os.environ["TFHUB_CACHE_DIR"] = "tf_cache"
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Begin demo
continueDemo = True

while continueDemo:
    # Ask user for sentence input
    user_input = input("Enter sentence: ")

    # Pre-processing
    user_input = user_input.lower()
    user_input = user_input.encode("ascii", "ignore")
    user_input = user_input.decode()
    clean_text = ""
    for ch in user_input:
        if ch.isalnum() or ch.isspace():
            clean_text += ch
        if ch == "." or ch == "'":
            clean_text += ch
    clean_text = clean_text.strip()

    # Embedding
    text_input = np.array(use([clean_text]))

    # Getting Prediction
    prediction = loaded_model.predict(text_input)
    predicted_value = prediction[0][1] - prediction[0][0]
    sentiment = "Neutral"
    if predicted_value > 0:
        sentiment = "Positive"
    elif predicted_value < 0:
        sentiment = "Negative"

    # Save to csv file
    with open("demo_results.csv", "a") as file:
        file.write(f"{user_input},{prediction[0]},{predicted_value},{sentiment}\n")

    print("--------")
    print(f"Sentence: {user_input}\nPrediction: {predicted_value}\nSentiment: {sentiment}")
    print("--------")

    continueDemo = bool(int(input("Do you want to try another? Yes->1, No->0: ")))

print("Thank you, hope you liked the demo!")
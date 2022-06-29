import pandas as pd

labelled_dataset = pd.read_csv("labelled_data.csv", delimiter=", ")

#Dataset exploration
number_data_points = labelled_dataset.shape[0]
number_positive_sentiment = 0
number_negative_sentiment = 0
number_neutral_sentiment = 0

for value in labelled_dataset['sentiment']:
    if value < 0:
        number_negative_sentiment += 1
    elif value > 0:
        number_positive_sentiment += 1
    else:
        number_neutral_sentiment += 1

print(f"Positive: {number_positive_sentiment}, Neutral: {number_neutral_sentiment}, Negative: {number_negative_sentiment}")
print(f"Total: {number_data_points}")
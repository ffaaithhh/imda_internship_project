import csv
import sentiment_labeller as sl

# Open the csv
clean_csv = open('cleaned_data.csv', 'r', encoding='utf-8')
label_csv = open('labelled_data.csv', 'w', encoding='utf-8')

csv_reader = csv.reader(clean_csv)

for row in csv_reader:
    if len(row) == 0:
        continue

    text = "".join(row)

    if len(text) == 0 or text.isspace():
        continue

    label_csv.write(str(sl.ultimate_sentiment(text)) + ", ")
    label_csv.write(text + "\n")


# Close the csv
clean_csv.close()
label_csv.close()

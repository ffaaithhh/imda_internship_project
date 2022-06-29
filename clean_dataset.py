import csv
import re

# Access the csv files
raw_file = open('raw_data.csv', 'r', encoding="utf-8")
clean_file = open('cleaned_data.csv', 'w', encoding='utf-8')

# Reading the file
reader_object = csv.reader(raw_file)

# No-no list for first word in the sentence
ban_list = ['/r/singapore', '[deleted]'] # the first one is a sticky thread title, the second are deleted posts

# Reading the file line by line
for line in reader_object:
    text = ",".join(line)
    sentence = text.split()
    
    clean_text = ""

    # Check not empty and not in ban list
    if  len(text) != 0 and len(sentence) != 0 and sentence[0] not in ban_list:

        # remove url
        text = re.sub(r'http\S+', '', text)

        # lowercase
        text = text.lower()

        # remove unicode
        text = text.encode("ascii", "ignore")
        text = text.decode()

        # remove special characters
        for ch in text:
            if ch.isalnum() or ch.isspace():
                clean_text += ch
            if ch == "." or ch == "'":
                clean_text += ch

        # remove extra spaces
        clean_text = clean_text.strip()

        # add to the cleaned csv
        if len(clean_text) != 0:
            clean_file.write(clean_text + "\n")

# Close the csv files
raw_file.close()
clean_file.close()
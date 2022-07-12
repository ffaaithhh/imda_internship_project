import pandas as pd
import dataframe_image as dfi
import pandas as pd

labelled_dataset = pd.read_csv("labelled_data.csv", delimiter=", ")

# Dataset exploration counting the positive and negative sentiments
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

# SENSITIVE ATTRIBUTES
gender = ['woman', 'women', 'girl', 'girls', 'girlfriend', 'mother', 'daughter', 'grandmother', 'aunt', 'female', 'feminine', 'maternal', 'man', 'men', 'boy', 'boys', 'boyfriend', 'father', 'son', 'grandfather', 'uncle', 'male', 'masculine', 'paternal',
          'agender', 'androgyne', 'bigender', 'butch', 'cisgender', 'gender expansive', 'genderfluid', 'gender outlaw', 'genderqueer', 'masucline of center', 'nonbinary', 'omnigender', 'polygender', 'pangender', 'transgender', 'trans', 'two spirit', 'gender neutral']
race = ['indian', 'malay', 'chinese', 'eurasian', 'peranakan', 'indian people', 'malay people', 'chinese people', 'eurasian people',
        'peranakan people', 'black', 'white', 'yellow', 'brown', 'black people', 'white people', 'yellow people', 'brown people']
sexual_orientation = ['homosexual', 'asexual', 'aromantic', 'heterosexual',
                      'straight', 'demisexual', 'gay', 'lesbian', 'bisexual', 'pansexual', 'queer']
nationality = ['local', 'foreigner', 'native', 'singaporean', 'malaysian', 'indonesian', 'indian', 'bangladeshis', 'bruneian', 'cambodian', 'laotian', 'burmese', 'filipino', 'vietnamese',
               'thai', 'korean', 'south korean', 'north korean', 'chinese', 'japanese', 'american', 'british', 'australian', 'french', 'mexican', 'african', 'russian', 'german', 'iranian', 'iraqis']
country = ['singapore', 'malaysia', 'indonesia', 'india', 'bangladesh', 'brunei', 'cambodia', 'lao', 'myanmar', 'burma', 'philippines', 'vietnam', 'thailand', 'korea',
           'south korea', 'north korea', 'china', 'japan', 'america', 'britian', 'australia', 'france', 'mexico', 'nigeria', 'south africa', 'egypt', 'russia', 'germany', 'iran', 'iraq']
religion = ['christianity', 'islam', 'taoism', 'hinduism',
            'buddhism', 'sikhism', 'shinto', 'judaism']
religion_identity = ['christian', 'muslim', 'taoist', 'hindu',
                     'freethinker', 'buddhist', 'sikhs', 'shintoism', 'jew']
marital_status = ['single', 'married', 'divorced', 'widowed', 'separated']
child_status = ['childfree', 'childlessness', 'childless', 'mother', 'father',
                'child', 'children', 'baby', 'toddler', 'teenager', 'adult', 'guardian']
age = ['young', 'old', 'middle-aged', 'senility', "aging", "mother", "grandmother", "father", "grandfather",
       "older", "younger", "retiring", "retiree", "youth", "elderly", "old people", "young people"]
ability_status = ['able-bodied', 'handicap', 'disability', 'differently abled', 'challenged', 'disabled people', 'people with disabilities', 'high-functioning', 'low-functioning', 'neurodiverse', 'neurodivergent', 'special needs', 'functional needs']
other_status = ['alcoholic', 'addiction', 'junkie', 'addict', 'smoker']
socio_ecconomic_status = ['low income', 'middle income', 'high income', 'working class', 'middle class', 'upper class', 'poor', 'rich', 'wealthy', 'poverty', 'renter', 'home owner']
education_status = ['doctoral degree', 'professional degree', "master's degree", "bachelor's degree", 'associate degree', 'no degree', 'high school graduate', 'junior college', 'polytechnic', 'diploma', 'primary school', 'secondary school', 'college', 'university']
gauge_words = ['good', 'bad', 'virtuous', 'evil', 'moral', 'immoral', 'happy', 'angry', 'sad', 'confused', 'awkward', 'beautiful', 'ugly', 'love', 'hate']

attributes = {"gender" : gender, "race" : race, "sexual_orientation" : sexual_orientation, "nationality" : nationality, "religion" : religion, "religion_identity" : religion_identity, "marital_status" : marital_status, "child_status" : child_status, "age" : age, "ability_status" : ability_status, "other_status" : other_status, "socio_economic_status" : socio_ecconomic_status, "education_status" : education_status, "gauge_words" : gauge_words, "country" : country}

# Dataset exploration counting sentitive attribute words
for attr in attributes.keys():
    count = []
    # For every word in the array
    for word in attributes[attr]:
        wordCount = 0
        # For every sentence
        for sentence in labelled_dataset['text']:
            # If the text is one word
            if len(word) == 1:
                words = sentence.split()
                for w in words:
                    if w == word:
                        wordCount += 1
            else:
                # More than one word
                if word in sentence:
                    wordCount += 1

        count.append(wordCount)

    data = {"text" : attributes[attr], "count" : count}
    df = pd.DataFrame(data)
    dfi.export(df, f"Sensitive attribute {attr} count.png")
    print(f"Completed aggregation for {attr}!")
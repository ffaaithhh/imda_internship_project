import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.models import model_from_json
from encoder import encode_text
import dataframe_image as dfi
import pandas as pd

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

for attr in attributes.keys():
    # Encode the text
    encoded = encode_text(attributes[attr])
    
    # Load the json and create the model
    json_file = open('model_num.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights("model_num.h5")
    loaded_model = load_model("model_num.hdf5")

    # Model's predictions
    pred_sentiments = loaded_model.predict(encoded)
    sent_classes = np.argmax(pred_sentiments, axis=1)

    # Store the data
    data = {"text" : attributes[attr], "faith's biased model" : sent_classes}

    # Save as image
    df = pd.DataFrame(data)
    dfi.export(df, f"biased model analysis on {attr}.png")

    print(f"Completed testing for {attr}")

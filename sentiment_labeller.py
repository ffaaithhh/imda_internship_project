from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
import dataframe_image as dfi

classifier = TextClassifier.load('en-sentiment')


def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

# This is NLTK's built-in pre-trained sentiment analyzer
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


def flair_sentiment(text, classifier):
    sentence = Sentence(text)
    classifier.predict(sentence)
    sentiment = sentence.labels[0].score

    if sentence.labels[0].value == "NEGATIVE":
        sentiment = -sentiment

    return sentiment


def ultimate_sentiment(text):
    total_sentiment = textblob_sentiment(
        text) + vader_sentiment(text) + flair_sentiment(text, classifier)
    sentiment = (total_sentiment) / 3
    return sentiment


def print_sentiment_image(text, term):
    data = {}
    data["text"] = text
    textblob = []
    vader = []
    flair = []
    total = []

    for t in text:
        textblob.append(textblob_sentiment(t))
        vader.append(vader_sentiment(t))
        flair.append(flair_sentiment(t, classifier))
        total.append(ultimate_sentiment(t))

    data['textblob'] = textblob
    data['vader'] = vader
    data['flair'] = flair
    data['total'] = total

    df = pd.DataFrame(data)
    dfi.export(df, f"sentiment analysis on {term}.png")

    return True


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


#print_sentiment_image(gauge_words, "gauge words")

#print(flair_sentiment("", classifier))
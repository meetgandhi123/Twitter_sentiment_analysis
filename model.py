import re
import pickle
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

def text_processing(tweet):    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
        
    return normalization(no_punc_tweet)

def train():
    train_tweets = pd.read_csv('tweet.csv')
    train_tweets=train_tweets.dropna()
    train_tweets = train_tweets[['label','tweet']]
    for i in range(0,len(train_tweets['label'])):
        train_tweets['label'][i]=int(train_tweets['label'][i])

    train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)

    train_tweets[train_tweets['label']==1].drop('tweet',axis=1).head()

    X = train_tweets['tweet']
    y = train_tweets['label']

    from sklearn.model_selection import train_test_split
    msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)

    #Machine Learning Pipeline
    pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train,label_train)

    # save the model to disk
    filename = 'finalized_model.pkl'
    pickle.dump(pipeline, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(msg_test, label_test)
    print("Testing accuracy_score:",result)

try:
    f = open("finalized_model.pkl")
except IOError:
    print("Training the model")
    train()
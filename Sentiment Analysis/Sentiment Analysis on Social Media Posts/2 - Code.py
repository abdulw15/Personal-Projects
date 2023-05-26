!pip install pandas
!pip install numpy
!pip install sklearn
!pip install nltk
!pip install seaborn
!pip install matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

tweets = pd.read_csv('tweets.csv')

# Drop unnecessary columns
tweets = tweets.drop(['id', 'user'], axis=1)

# Replace target labels with numerical values
tweets = tweets.replace({'target': {'negative': 0, 'positive': 1}})

# Remove stopwords and stem words
stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

def process_tweet(tweet):
    # Remove punctuation and special characters
    tweet = re.sub(r'\W+', ' ', tweet)
    
    # Tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    # Remove stopwords and stem words
    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # Remove stopwords
                word not in string.punctuation):  # Remove punctuation
            stem_word = stemmer.stem(word)  # Stemming word
            tweet_clean.append(stem_word)
            
    return tweet_clean

# Apply process_tweet function to each tweet in the dataset
tweets['text'] = tweets['text'].apply(process_tweet)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweets['text'], tweets['target'], test_size=0.2, random_state=42)

# Create bag of words model
vectorizer = CountVectorizer(analyzer=process_tweet)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on test set
y_pred = nb_classifier.predict(X_test)

# Evaluate performance of classifier
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


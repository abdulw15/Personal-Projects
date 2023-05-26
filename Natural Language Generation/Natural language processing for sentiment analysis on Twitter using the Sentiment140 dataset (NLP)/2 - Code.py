# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB

# Load the Sentiment140 dataset
df = pd.read_csv('Sentiment140.csv', encoding='ISO-8859-1', header=None)

# Rename the columns
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Remove unnecessary columns
df = df[['target', 'text']]

# Map the target values to binary labels (0=negative, 4=positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# Define a function to preprocess the tweet text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_tweet_text(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)

    # Remove user mentions and hashtags
    tweet = re.sub(r'@\w+|#\w+', '', tweet)

    # Remove special characters and numbers
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenize the tweet
    tokens = nltk.word_tokenize(tweet)

    # Remove stop words and stem the remaining words
    tokens = [ps.stem(token) for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    tweet = ' '.join(tokens)

    return tweet

# Preprocess the tweet text
df['text'] = df['text'].apply(preprocess_tweet_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Vectorize the preprocessed text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_vec)

# Evaluate the performance of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the dataset
data = pd.read_csv('Tweets.csv')

# Data preprocessing
# Remove irrelevant columns
data = data[['text', 'airline_sentiment']]

# Remove neutral tweets
data = data[data['airline_sentiment'] != 'neutral']

# Remove twitter handles, numbers, and special characters from tweet text
def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

data['text'] = data['text'].apply(preprocess_text)

# Tokenize tweet text
def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

data['tokens'] = data['text'].apply(tokenize_text)

# Sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['text'].apply(get_sentiment)

# Topic modeling
vectorizer = CountVectorizer(tokenizer=tokenize_text, stop_words='english', max_df=0.95, min_df=2, max_features=1000)
X = vectorizer.fit_transform(data['text'])
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)
for i, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]]
    print(f"Topic {i}: {', '.join(top_words)}")

# Visualization
sns.countplot(x='airline_sentiment', hue='sentiment', data=data)
plt.show()


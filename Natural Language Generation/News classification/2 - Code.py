# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Loading the dataset
df = pd.read_csv('bbc_news.csv')

# Printing the first 5 rows of the dataset
print(df.head())

# Printing the shape of the dataset
print(df.shape)

# Checking for null values
print(df.isnull().sum())

# Checking the distribution of the target variable
print(df['category'].value_counts())

# Creating the feature matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Creating the target vector
y = df['category']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiating the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Fitting the classifier to the training data
clf.fit(X_train, y_train)

# Predicting the target variable for the test data
y_pred = clf.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))

# Saving the model
import joblib

filename = 'bbc_news_model.sav'
joblib.dump(clf, filename)

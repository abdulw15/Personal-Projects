# Section 1: Importing Libraries and Loading Data
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Explore the dataset
print(df.shape)
print(df.head())
print(df.tail())

# Section 2: Data Preprocessing and Cleaning
# Remove unwanted columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Convert the target variable into binary form (0 and 1)
df['label'] = np.where(df['label']=='spam', 1, 0)

# Convert all text to lower case
df['text'] = df['text'].apply(lambda x: x.lower())

# Remove punctuation and special characters
df['text'] = df['text'].str.replace('[^\w\s]', '')

# Tokenize the text data into individual words
df['text'] = df['text'].apply(lambda x: x.split())

# Remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

# Apply stemming or lemmatization to the words
from nltk.stem import PorterStemmer

def stem_words(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

df['text'] = df['text'].apply(lambda x: stem_words(x))

# Section 3: Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'].apply(lambda x: ' '.join(x)))

y = df['label']

# Section 4: Train the model

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages['message'], messages['label'], test_size=0.2, random_state=42)

# Vectorize the messages using TF-IDF vectorizer
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# Train a Linear Support Vector Classifier model
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# Evaluate the model on the testing set
y_pred = svm.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Model Optimization

# We will use the GridSearchCV function to perform hyperparameter tuning. GridSearchCV allows us to define a grid of hyperparameters and then search for the optimal combination of hyperparameters by exhaustively trying every possible combination.

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters and the corresponding score
print('Best parameters: ', grid_search.best_params_)
print('Best score: ', grid_search.best_score_)

#Model Evaluation 

# Predict the labels of the test set
y_pred = grid_search.predict(X_test_tfidf)

# Print the accuracy score of the optimized model
print('Accuracy score: ', accuracy_score(y_test, y_pred))

# Print the confusion matrix of the optimized model
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion matrix: \n', conf_mat)

# Print the classification report of the optimized model
class_report = classification_report(y_test, y_pred)
print('Classification report: \n', class_report)


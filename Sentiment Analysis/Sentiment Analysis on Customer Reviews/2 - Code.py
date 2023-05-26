# Import necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('customer_reviews.csv', encoding='latin-1')
data = data[['text', 'label']]

# Data preprocessing
data['text'] = data['text'].apply(str)
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
data['text'] = data['text'].apply(lambda x: ' '.join(x))

# Feature extraction
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(data['text'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)

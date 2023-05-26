import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('document_classification_data.csv')

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Preprocessing
vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(train_data['text'])
test_features = vectorizer.transform(test_data['text'])
train_labels = train_data['category']
test_labels = test_data['category']


# Create and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

# Make predictions on the test set
test_predictions = classifier.predict(test_features)

# Calculate accuracy score
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy:", accuracy)

# Calculate confusion matrix and classification report
cm = confusion_matrix(test_labels, test_predictions)
cr = classification_report(test_labels, test_predictions)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)


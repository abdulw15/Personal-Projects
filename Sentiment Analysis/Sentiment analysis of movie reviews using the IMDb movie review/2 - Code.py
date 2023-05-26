# Section 1: Importing Required Libraries and Loading the Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the IMDb movie review dataset
data = pd.read_csv('IMDb_Dataset.csv')

# Section 2: Data Preprocessing
# Drop any rows with missing values
data.dropna(inplace=True)

# Map sentiment to binary values
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Section 3: Splitting the Dataset into Training and Testing Sets
# Split the data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.3)

# Section 4: Feature Extraction using TF-IDF Vectorizer
# Convert the text into feature vectors using the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Section 5: Training the Logistic Regression Model
# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Section 6: Evaluating the Model
# Calculate the accuracy of the model on the training and testing sets
training_accuracy = model.score(X_train, y_train)
testing_accuracy = model.score(X_test, y_test)
print(f'Training Accuracy: {training_accuracy}')
print(f'Testing Accuracy: {testing_accuracy}')


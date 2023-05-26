import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('cardio_data.csv', delimiter=';')

# Drop irrelevant columns
data.drop(['id'], axis=1, inplace=True)

# Convert age from days to years
data['age'] = (data['age'] / 365).round().astype(int)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['gender', 'cholesterol', 'gluc'])

# Split into features and labels
X = data.drop(['cardio'], axis=1)
y = data['cardio']

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import pickle

# Save the model
with open('cardio_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load the model and make predictions on new data
with open('cardio_model.pkl', 'rb') as f:
    model = pickle.load(f)

new_data = pd.DataFrame({'age': [50], 'height': [165], 'weight': [75], 'ap_hi': [120], 'ap_lo': [80], 'cholesterol_1': [0],
                         'cholesterol_2': [1], 'cholesterol_3': [0], 'gluc_1': [1], 'gluc_2': [0], 'gluc_3': [0],
                         'gender_1': [0], 'gender_2': [1]})

new_data = pd.get_dummies(new_data, columns=['gender', 'cholesterol', 'gluc'])
new_data = scaler.transform(new_data)

prediction = model.predict(new_data)
print("Prediction:", prediction)


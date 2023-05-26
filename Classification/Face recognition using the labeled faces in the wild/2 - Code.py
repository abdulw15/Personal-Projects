import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os

# Load the dataset from Kaggle
lfw_dir = 'lfw-deepfunneled'
people = [person for person in os.listdir(lfw_dir)]

X = []
y = []

for person in people:
    person_dir = os.path.join(lfw_dir, person)
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)
        y.append(person)

# Preprocess the images using OpenCV
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

X_processed = []

for img in X:
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (50, 50))
        X_processed.append(face_img)
        
X_processed = np.array(X_processed)
y = np.array(y)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Extract features using PCA
pca = PCA(n_components=100, whiten=True)
X_train_pca = pca.fit_transform(X_train.reshape((X_train.shape[0], -1)))
X_test_pca = pca.transform(X_test.reshape((X_test.shape[0], -1)))

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

# Evaluate the performance of the model on the test set
y_pred = knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#Section 1: Importing Required Libraries and Loading Dataset

import pandas as pd
import numpy as np
import json
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_json('train-v2.0.json')

#Section 2: Preprocessing Dataset


# Define a function to preprocess text
def preprocess_text(text):
    # Remove punctuations, numbers and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words and lemmatize the words
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join the preprocessed tokens
    preprocessed_text = ' '.join(preprocessed_tokens)
    return preprocessed_text

# Preprocess the context and question text
contexts = []
questions = []
for topic in data['data']:
    for paragraph in topic['paragraphs']:
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                context = preprocess_text(paragraph['context'])
                question = preprocess_text(qa['question'])
                contexts.append(context)
                questions.append(question)

#Section 3: Creating TF-IDF Vectors

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the context data
context_vectors = vectorizer.fit_transform(contexts)

# Transform the question data
question_vectors = vectorizer.transform(questions)

#Section 4: Finding Relevant Passages

# Define a function to find the relevant passage
def find_relevant_passage(question, context_vectors, question_vectors, contexts):
    # Transform the question
    question_vector = vectorizer.transform([question])

    # Calculate cosine similarity between the question and context vectors
    similarities = cosine_similarity(question_vector, context_vectors)

    # Find the index of the most similar context vector
    idx = np.argmax(similarities)

    # Get the corresponding context text
    context = contexts[idx]

    # Tokenize the context text into sentences
    sentences = sent_tokenize(context)

    # Calculate cosine similarity between the question and each sentence vector
    sentence_vectors = vectorizer.transform(sentences)
    sentence_similarities = cosine_similarity(question_vector, sentence_vectors)

    # Find the index of the most similar sentence vector
    sentence_idx = np.argmax(sentence_similarities)

    # Get the corresponding sentence text
    relevant_passage = sentences[sentence_idx]

    return relevant_passage



#Section 5: User Interface

# Define a function for the user interface
def user_interface():
    # Get the user input
    question = input("Enter your question: ")

    # Find the relevant passage
    relevant_passage = find_relevant_passage(question, context_vectors, question_vectors, contexts)

    # Display the relevant passage
    print("Relevant Passage: ", relevant_passage)

# Run the user interface
while True:
    user_interface()

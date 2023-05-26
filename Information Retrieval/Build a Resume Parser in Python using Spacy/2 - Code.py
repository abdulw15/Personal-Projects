#1. Import required libraries
import pandas as pd

#. Load the dataset
df = pd.read_csv('resume_dataset.csv')

#Data Cleaning and Preprocessing
#Convert text to lowercase
df['resume_text'] = df['resume_text'].str.lower()

#Remove unwanted characters
df['resume_text'] = df['resume_text'].str.replace('\r', '')
df['resume_text'] = df['resume_text'].str.replace('\n', '')
df['resume_text'] = df['resume_text'].str.replace('\t', '')

#Remove stop words
import spacy

nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

df['resume_text'] = df['resume_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

df['doc'] = df['resume_text'].apply(nlp)

#Tokenize the text and assign part-of-speech tags to each token using Spacy
import random
import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load('en_core_web_sm')
ner = nlp.create_pipe('ner')
nlp.add_pipe(ner)

TRAIN_DATA = [('John Smith is a software engineer with 10 years of experience.', {'entities': [(0, 10, 'PERSON'), (22, 38, 'JOB_TITLE'), (42, 44, 'YEARS_OF_EXPERIENCE')]}), ...]

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for iteration in range(100):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)

#Train a named entity recognition (NER) model on the preprocessed data

import random
import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load('en_core_web_sm')
ner = nlp.create_pipe('ner')
nlp.add_pipe(ner)

TRAIN_DATA = [('John Smith is a software engineer with 10 years of experience.', {'entities': [(0, 10, 'PERSON'), (22, 38, 'JOB_TITLE'), (42, 44, 'YEARS_OF_EXPERIENCE')]}), ...]

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for iteration in range(100):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)

#Identify and extract important entities like candidate name, email address, phone number, skills, experience, education, etc. from the resumes

def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

df['entities'] = df['doc'].apply(extract_entities)

#Use Spacy's dependency parsing capabilities to extract structured information like job titles, companies, dates, and locations from the resumes
def extract_info(doc):
    info = {}
    for ent in doc.ents:
        if ent.label_ == 'JOB_TITLE':
            info['job_title'] = ent.text
        elif ent.label_ == 'ORG':
            info['company'] = ent.text
        elif ent.label_ == 'DATE':
            info['date'] = ent.text
        elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
            if 'location' in info:
                info['location'].append(ent.text)
            else:
                info['location'] = [ent.text]
    return info

# Structure the extracted entities and information in a structured format like JSON or CSV for further analysis.

import json

resumes = []

for resume in resume_files:
    with open(resume, 'r') as f:
        text = f.read()
        doc = nlp(text)
        info = extract_info(doc)
        info['resume'] = resume
        resumes.append(info)

with open('resume_info.json', 'w') as f:
    json.dump(resumes, f, indent=4)


#def evaluate_parser(model, test_data):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for resume, labels in test_data:
        doc = model(resume)
        predicted_labels = extract_info(doc)
        
        for label in labels:
            if label in predicted_labels:
                true_positives += 1
            else:
                false_negatives += 1
                
        for label in predicted_labels:
            if label not in labels:
                false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

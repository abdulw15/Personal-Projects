import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    article = ""
    with open(file_name, "r") as f:
        for line in f.readlines():
            article += line
    article = article.replace("\n", " ")
    return article

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
        
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    
    sentences = nltk.sent_tokenize(read_article(file_name))
    
    sentence_vectors = np.zeros((len(sentences), 300))
    
    for i in range(len(sentences)):
        sentence_vectors[i] = np.mean([embeddings_dict.get(w, np.zeros((300,))) for w in sentences[i].split()], axis=0)
    
    sim_mat = np.zeros([len(sentences), len(sentences)])
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0,0]
    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    return ". ".join(summarize_text)

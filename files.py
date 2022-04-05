import os
import json
import math
import datetime
import time
import spacy
import nltk
import pyLDAvis
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords

# Data
data_folder = 'Downloaded_PlaneReports'
mdl_file = 'trained-mdl'

def preprocess(start_date=None, end_date=None):

    docs = []           # List of documents
    vocab = Counter()   # Counter keys are terms, values are term frequencies

    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    en = spacy.load('en_core_web_sm')

    # Convert dates
    if start_date != None: start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if end_date != None: end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    if end_date == None: end_date = datetime.datetime.now()

    # Words to remove from vocab
    remove_words = ['pilot', 'factor', 'contributing', 'aircraft', 'airplane', 'plane', 'due', 'could', 'would', 'resulted', 'failure', 'landing',
                    'loss', 'maintain', 'control', 'directional', 'flight', 'reasons', 'determined', 'runway']

    # Extract all probable cause lines from json files
    files = os.listdir(data_folder)
    for file in files:
        f = open(os.path.join(data_folder, file), encoding='utf8')
        file_contents = json.load(f)
        for crash in file_contents:
            inDateRange = False
            for tag in crash.keys():
                if tag == 'cm_probableCause' and crash[tag] != None:
                    tokens = word_tokenize(crash[tag])
                    if (len(tokens) > 0):
                        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
                        tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english') and len(token) > 2]
                        tokens = [token for token in tokens if token not in remove_words]
                elif tag == 'cm_eventDate':
                    event_date = crash[tag][0:10]
                    event_date = datetime.datetime.strptime(event_date, '%Y-%m-%d')
                    if (start_date == None or (start_date <= event_date <= end_date)): # Check date range
                        inDateRange = True

            if inDateRange == True and len(tokens) > 0:
                # Add tokens to docs list and update vocab numbers and data
                docs.append(tokens)
                vocab.update(tokens)

    return docs, vocab

def prepare_vis_data(mdl, docs, vocab, K):
    def prepare_data():
        nonlocal mdl, docs, K
        doc_topic_dists = [mdl.score(doc) for doc in docs]
        doc_lengths = [len(doc) for doc in docs]
        doc_topic_dists2 = [[v if not math.isnan(v) else 1/K for v in d]
                            for d in doc_topic_dists]
        doc_topic_dists2 = [d if sum(d) > 0 else [1/K]*K for d in
                            doc_topic_dists]
        matrix = []
        for cluster in mdl.cluster_word_distribution:
            total = sum(cluster.values())
            row = [cluster.get(k, 0) / total for k in vocab]
            matrix.append(row)
        return matrix, doc_topic_dists2, doc_lengths

    out = pyLDAvis.prepare(*prepare_data(), vocab.keys(), vocab.values(),
                           sort_topics=False, mds='mmds')
    return out


import os
import json
import math
import spacy
import nltk
import pyLDAvis
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

# Data
data_folder = 'Downloaded_PlaneReports'
mdl_file = 'trained-mdl.pkl'

def preprocess():

    docs = []           # List of documents
    vocab = Counter()   # Counter keys are terms, values are term frequencies

    nltk.download('punkt')
    nltk.download('stopwords')
    en = spacy.load('en_core_web_sm')

    # Words to remove from vocab
    remove_words = ["'s", 'pilot', 'airplane', 'plane', 'due', 'could', 'resulted', 'failure', 'landing', 'loss',
                    'maintain', 'control', 'directional', 'flight', 'reasons', 'determined', 'runway']

    # Extract all probable cause lines from json files
    files = os.listdir(data_folder)
    for file in files:
        f = open(os.path.join(data_folder, file), encoding='utf8')
        file_contents = json.load(f)
        for crash in file_contents:
            for tag in crash.keys():
                if tag == 'cm_probableCause':
                    tokens = word_tokenize(crash[tag])
                    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')
                              and len(token) > 1 and token not in remove_words]

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


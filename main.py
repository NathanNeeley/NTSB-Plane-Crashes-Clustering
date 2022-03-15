import os
import pickle
import numpy as np
import files
import pyLDAvis
from files import preprocess
from files import prepare_vis_data
from GSDMM import GSDMM

if __name__ == '__main__':
    data, mdl, docs, vocab = None, None, None, None

    # Check if there's a saved model
    if os.path.exists(files.mdl_file):
        print('Reading from the trained model file.')
        with open(files.mdl_file, 'rb') as f:
            data = pickle.load(f)
            mdl = data['mdl']
            docs = data['docs']
            vocab = data['vocab']
    else:
        # TODO -- run web crawler

        # Extract data from json files, tokenize it, and remove non-descriptive words
        docs, vocab = preprocess()

        # Fit data and group into clusters
        mdl = GSDMM()
        model = mdl.fit(docs, len(vocab))
        mdl = GSDMM.from_data(K=10, alpha=0.1, beta=0.1, D=len(docs), vocab_size=len(vocab), cluster_doc_count=mdl.cluster_doc_count, cluster_word_count=mdl.cluster_word_count, cluster_word_distribution=mdl.cluster_word_distribution)

        # Save model
        with open(files.mdl_file, 'wb') as f:
            data = {
                'mdl': mdl,
                'docs': docs,
                'vocab': vocab
            }
            pickle.dump(data, f)

    # #docs per cluster
    print('#docs per cluster: ', mdl.cluster_doc_count)

    # Order clusters and print out
    doc_count = np.array(mdl.cluster_doc_count)
    top_index = doc_count.argsort()[-10:][::-1]
    for cluster in top_index:
        sort_dicts = sorted(mdl.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:10]
        print("Cluster %s : %s" % (cluster, [i[0] for i in sort_dicts]))

    # Visualize vocab and clusters
    vis_data = prepare_vis_data(mdl, docs, vocab, K=10)
    pyLDAvis.save_html(vis_data, 'clusters-vis.html')

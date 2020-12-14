import math
import sys
import os
import numpy as np
import itertools
from sklearn import metrics
from utils import flatten, find_cluster_indices, cluster_documents

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils.dataset_constants import DATASETS
from dataset_utils.dataset_load import load_test_set, load_train_set

DECAY_THRESHOLD = 100
IDF_THRESHOLD = 3
KEEP_WORDS = 10
SIM_THRESHOLD = 50

forbidden_symbols = {
    ".",
    "!",
    "?",
    "'",
    "-",
    ";",
    ":",
    ",",
    "--",
    "(",
    ")",
    "'",
    "''",
    "—",
    "”",
    "’",
    "“",
    "&",
    "``",
    "/",
    "...",
    "s",
    "nt",
    "re",
    "ve",
    "d",
    "t",
    "@",
    "#",
    "$"
}


def create_clusters(documents):
    for document in documents:
        document.text = []
        for sentence in document.original:
            words = [word.lower() for word in sentence]
            words = [word for word in words if word not in forbidden_symbols]
            document.text.append(words)
    num_docs = len(documents)
    term_doc_frequency = {}
    # calculate term frequency per document
    for document in documents:
        words = flatten(document.text)
        words = words[:DECAY_THRESHOLD]
        doc_corpus = {}
        for word in words:
            if word in doc_corpus:
                doc_corpus[word] += 1
            else:
                doc_corpus[word] = 1
        document.corpus = doc_corpus
    # calculate number of documents where each term appears
    for document in documents:
        for word in document.corpus:
            if word in term_doc_frequency:
                term_doc_frequency[word] += 1
            else:
                term_doc_frequency[word] = 1
    # calculate the idf for each term, and ignore if it is below the threshold
    term_idf = {}
    for term in term_doc_frequency:
        idf = math.log(num_docs / (1 + term_doc_frequency[term]))
        if idf >= IDF_THRESHOLD:
            term_idf[term] = idf
    # generate the tf_idf vector for each document
    for document in documents:
        document.tf_idf_vector = generate_tf_idf_vector(document, term_idf)

    clusters = [Cluster(documents[0], term_idf)]

    for document in documents[1:]:
        clusters = calc_similarity(clusters, document, term_idf)

    return clusters


def calc_similarity(clusters, document, term_idf):
    # calculate the similarity between a document and a cluster. If the similarity is greather than SIM_THRESHOLD, add it to the cluster, otherwise create a new cluster
    max_sim = 0
    best_cluster = None
    for cluster in clusters:
        sim = 0
        for term in cluster.centroid:
            sim += (
                cluster.centroid[term] * document.tf_idf_vector[term]
                if term in document.tf_idf_vector
                else 0
            )
        if sim > max_sim:
            max_sim = sim
            best_cluster = cluster
    if max_sim > SIM_THRESHOLD:
        best_cluster.add_document(document)
    else:
        new_cluster = Cluster(document, term_idf)
        clusters.append(new_cluster)
    return clusters


def generate_tf_idf_vector(document, term_idf):
    vector = {}
    for term in document.corpus:
        if term in term_idf:
            term_frequency = math.log(1 + document.corpus[term])
            idf = term_idf[term]
            vector[term] = term_frequency * idf
    return vector


def tf_idf(document, term, term_idf):
    term_frequency = math.log(1 + (document.corpus[term] / document.word_count))
    inverse_document_frequency = term_idf[term]

    return term_frequency * inverse_document_frequency


class Cluster:
    def __init__(self, document, term_idf):
        self.documents = [document]
        self.centroid = {}
        self.term_idf = term_idf
        self._update_centroid()

    def add_document(self, document):
        self.documents.append(document)
        self._update_centroid()

    def _update_centroid(self):
        centroid = {}
        # get the KEEP_WORDS terms whose score is the highest, where score = average frequency over documents * idf
        avg_freq = {}
        for document in self.documents:
            avg_freq.update(document.corpus)
        terms_tdidf = {}
        for word in avg_freq:
            if word in self.term_idf:
                terms_tdidf[word] = (
                    avg_freq[word] / len(self.documents)
                ) * self.term_idf[word]

        for i in range(KEEP_WORDS):
            if terms_tdidf.keys():
                max_term = max(terms_tdidf, key=lambda key: terms_tdidf[key])
                centroid[max_term] = terms_tdidf[max_term]
                del terms_tdidf[max_term]
            else:
                break
        self.centroid = centroid


def main():
    dataset = load_test_set("multi_news", True)
    cluster_idxs = find_cluster_indices(dataset)
    true_clusters = cluster_documents(cluster_idxs, dataset)
    docs = []
    for cluster in true_clusters:
        for doc in true_clusters[cluster]:
            docs.append(doc)
    clusters = create_clusters(docs)

    cluster_dict = {}
    for i in range(len(clusters)):
        for doc in clusters[i].documents:
            cluster_dict[doc] = i

    true_cluster_dict = {}
    for i in range(len(true_clusters) + 1):
        if i == 136:
            pass
        else:
            for doc in true_clusters[i]:
                true_cluster_dict[doc] = i

    t_pos = 0
    t_neg = 0
    f_pos = 0
    f_neg = 0
    for tup in itertools.combinations(docs, 2):
        doc1, doc2 = tup
        if (
            true_cluster_dict[doc1] == true_cluster_dict[doc2]
            and cluster_dict[doc1] == cluster_dict[doc2]
        ):
            t_pos += 1
        elif (
            true_cluster_dict[doc1] == true_cluster_dict[doc2]
            and cluster_dict[doc1] != cluster_dict[doc2]
        ):
            f_neg += 1
        elif (
            true_cluster_dict[doc1] != true_cluster_dict[doc2]
            and cluster_dict[doc1] != cluster_dict[doc2]
        ):
            t_neg += 1
        elif (
            true_cluster_dict[doc1] != true_cluster_dict[doc2]
            and cluster_dict[doc1] == cluster_dict[doc2]
        ):
            f_pos += 1
    
    y_true = []
    y_pred = []
    for doc in docs:
        y_true.append(true_cluster_dict[doc])
        y_pred.append(cluster_dict[doc])
    
    purity = get_purity(y_true, y_pred)
    print(f"{t_pos=}")
    print(f"{t_neg=}")
    print(f"{f_pos=}")
    print(f"{f_neg=}")
    print(f"{purity=}")

def get_purity(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == "__main__":
    main()

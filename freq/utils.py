import nltk
from typing import List, Dict
from collections import defaultdict

stopwords = nltk.corpus.stopwords.words("english")
stopwords.extend(
    [
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
    ]
)
flatten = lambda t: [item for sublist in t for item in sublist]


class Document:
    def __init__(
        self, text: List[List[str]], summary: List[List[str]], cluster_id, **kwargs
    ):
        self.original = text
        self.summary = summary
        self.meta_data = kwargs
        self.cluster_id = cluster_id
        self.text = []
        self.corpus = {}
        self.tf_idf_vector = {}


def clean_sentences(document):
    for sentence in document.original:
        words = [word.lower() for word in sentence]
        words = [word for word in words if word not in stopwords]
        document.text.append(words)
    return document


def find_cluster_indices(dataset: List[Dict]) -> Dict[int, List[int]]:
    clusters = defaultdict(list)
    for e_num, example in enumerate(dataset):
        clusters[example["cluster-id"]].append(e_num)

    return clusters


def cluster_documents(cluster_idx, dataset):
    clusters = {}
    for c_num, c_list in cluster_idx.items():
        for doc_idx in c_list:
            example = dataset[doc_idx]
            doc = clean_sentences(
                Document(example["input"], example["output"], c_num, **example["meta"])
            )
            if c_num in clusters:
                clusters[c_num].append(doc)
            else:
                clusters[c_num] = [doc]
    return clusters


def generate_prob_dict(documents):
    num_tokens = 0
    word_probs = {}
    for document in documents:
        for sentence in document.text:
            num_tokens += len(sentence)
            for word in sentence:
                if word in word_probs:
                    word_probs[word] += 1
                else:
                    word_probs[word] = 1

    for word in word_probs:
        word_probs[word] = word_probs[word] / num_tokens

    return word_probs

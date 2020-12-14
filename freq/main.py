import sys 
import os
import argparse
from tabulate import tabulate
from rouge import Rouge
from nltk.tokenize.treebank import TreebankWordDetokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils.dataset_constants import DATASETS
from dataset_utils.dataset_load import load_test_set, load_train_set
from utils import find_cluster_indices, cluster_documents
from sumbasic import generate_summary_basic
from SumBasicExtended import generate_summary_extended

def main(args):
    if args.debug:
        clusters = load_test_data(args)
    else:
        clusters = load_train_data(args)
    
    summaries = collect_summaries(args, clusters)
    true_summaries = []
    for cluster in clusters:
        true_summaries.append(clusters[cluster][0].summary)
    
    evaluate_results(true_summaries, summaries)

def evaluate_results(data, summaries):
    rouge = Rouge()
    hyps = []
    refs = []
    hyps = [rebuild_summary(hyp) for hyp in summaries]
    refs = [rebuild_summary(ref) for ref in data]
    rouge_scores = rouge.get_scores(hyps=hyps, refs=refs, avg=True)
    metric = []
    f = []
    p = []
    r = []
    for key, vals in rouge_scores.items():
        metric.append(key)
        f_val = vals["f"]
        f.append(f"{f_val:0.3f}")
        p_val = vals["p"]
        p.append(f"{p_val:0.3f}")
        r_val = vals["r"]
        r.append(f"{r_val:0.3f}")
    print(
        tabulate(
            zip(metric, f, p, r),
            headers=["Rogue Metric", "f-score", "p-score", "r-score"],
            tablefmt="latex",
        )
    )


def rebuild_summary(summary_tokens):
    detokenizer = TreebankWordDetokenizer()
    return " ".join(
        [detokenizer.detokenize(sentence_tokens) for sentence_tokens in summary_tokens]
    )


def collect_summaries(args, clusters):
    summaries = []
    if args.model == 1:
        for cluster in clusters:
            documents = clusters[cluster]
            summary = generate_summary_basic(documents, args.sentence)
            summaries.append(summary)
    elif args.model ==2:
        for cluster in clusters:
            documents = clusters[cluster]
            summary = generate_summary_extended(documents, args.sentence)
            summaries.append(summary)

    return summaries

def load_test_data(args):
    dataset = load_test_set(args.dataset, args.debug)
    cluster_idx = find_cluster_indices(dataset)
    clusters = cluster_documents(cluster_idx, dataset)
    return clusters

def load_train_data(args):
    dataset = load_train_set(args.dataset, args.debug)
    cluster_idx = find_cluster_indices(dataset)
    clusters = cluster_documents(cluster_idx, dataset)
    return clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model",
        "-m",
        type=int,
        required=True,
        help="Select the model to run: 1 for SumBasic, 2 for SumBasicExtended",
        choices=range(1, 3),
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Select the dataset to evaluate", 
        choices=DATASETS,
    )
    parser.add_argument(
        "--debug",
        "-de",
        required=False,
        action="store_true",
        help="Select if we should run a debug run",
        default=False,
    )
    parser.add_argument(
        "--sentence", "-s", type=int, help="Number of sentences to produce", default=3
    )
    args = parser.parse_args()
    main(args)

    
import argparse
import json
import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow_datasets as tfds
from progressbar import ProgressBar
from progressbar.widgets import ETA, Bar, Percentage

from dataset_utils.dataset_constants import DATASET_KEYS, DATASET_SPLITS, DATASETS
from dataset_utils.dataset_helpers import process_multi_news, process_text

DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DATASET_TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets-test"
)

convert_widgets = [
    "Extracted: ",
    Percentage(),
    " ",
    Bar(),
    " ",
    ETA(),
]


def get_dataset_folder(is_test: bool) -> str:
    return DATASET_TEST_FOLDER if is_test else DATASET_FOLDER


def install(args):
    # make datasets path if it does not exist
    os.makedirs(get_dataset_folder(args.test), exist_ok=True)

    # add tensorflow datasets here to ensure they are imported
    for dataset_name in DATASETS:
        for split in DATASET_SPLITS[dataset_name]:
            # load dataset
            print(f"Loading Dataset, {dataset_name}, on split, {split}")

            if dataset_name != "reddit_tifu/long":
                ds = tfds.load(dataset_name, split=split)
                if args.test:
                    examples = [example for example in ds.as_numpy_iterator()][:1000]
                else:
                    examples = [example for example in ds.as_numpy_iterator()]
            else:
                # straight jenk solution to reddit tifu not having 'test' data
                ds = tfds.load(dataset_name, split="train")
                if args.test:
                    if split == "train":
                        examples = [example for example in ds.as_numpy_iterator()][
                            :1000
                        ]
                    else:
                        ex = [example for example in ds.as_numpy_iterator()]
                        print(len(ex))
                        examples = [example for example in ds.as_numpy_iterator()][
                            35000:36000
                        ]
                else:
                    if split == "train":
                        examples = [example for example in ds.as_numpy_iterator()][
                            :35000
                        ]
                    else:
                        examples = [example for example in ds.as_numpy_iterator()][
                            35000:
                        ]

            print(f"Done Loading Dataset, {dataset_name}, on split, {split}\n")

            # go through examples and convert into a dictionary to be saved
            print(f"Converting Dataset, {dataset_name}, on split, {split}")

            # cluster id for Multi-News
            cluster_id = 0
            converted_examples = []
            convert_pbar = ProgressBar(widgets=convert_widgets)
            for example in convert_pbar(examples):
                # sets meta data if dataset has some
                meta = {}
                if DATASET_KEYS[dataset_name]["meta"] != []:
                    for meta_key in DATASET_KEYS[dataset_name]["meta"]:
                        meta[meta_key] = process_text(example[meta_key].decode("utf-8"))

                summary, stem_summary = process_text(
                    example[DATASET_KEYS[dataset_name]["output"]].decode("utf-8")
                )

                # ensures a summary exists
                if summary == []:
                    continue

                if dataset_name == "multi_news":
                    if cluster_id == 136:
                        cluster_id += 1
                        continue

                    multi_news_texts = process_multi_news(
                        example[DATASET_KEYS[dataset_name]["input"]].decode("utf-8")
                    )

                    for text, stem_text in multi_news_texts:
                        converted_examples.append(
                            {
                                "input": text,
                                "input-stem": stem_text,
                                "output": summary,
                                "output-stem": stem_summary,
                                "cluster-id": cluster_id,
                                "meta": meta,
                            }
                        )
                else:
                    text, stem_text = process_text(
                        example[DATASET_KEYS[dataset_name]["input"]].decode("utf-8")
                    )

                    converted_examples.append(
                        {
                            "input": text,
                            "input-stem": stem_text,
                            "output": summary,
                            "output-stem": stem_summary,
                            "cluster-id": cluster_id,
                            "meta": meta,
                        }
                    )

                cluster_id += 1

                if cluster_id > 55000 and dataset_name == "cnn_dailymail":
                    break

                if cluster_id > 28000 and dataset_name == "multi_news":
                    break

            print(f"Finished Converting Dataset, {dataset_name}, on split, {split}\n")

            # save data dictionary
            print(f"Saving Converted Dataset, {dataset_name}, on split, {split}")

            mod_dataset_name = dataset_name.replace("/", "_")
            with open(
                os.path.join(
                    get_dataset_folder(args.test), f"{mod_dataset_name}-{split}.json"
                ),
                "w",
            ) as f:
                json.dump(converted_examples, f)

            print(
                f"Finished Saving Converted Dataset, {dataset_name}, on split, {split}\n\n"
            )


# def split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install and process data")
    parser.add_argument(
        "--test",
        "-t",
        required=False,
        action="store_true",
        default=False,
        help="Generate test dataset",
    )
    args = parser.parse_args()
    install(args)

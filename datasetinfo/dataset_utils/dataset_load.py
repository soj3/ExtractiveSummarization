import os
import json
import sys
from typing import Dict, List

from dataset_utils.dataset_constants import DATASET_SPLITS, DATASETS

debug_run_DIR = "datasets-test"
FULL_RUN_DIR = "datasets"


def load_train_set(dataset_name: str, debug_run: bool):
    return _load_dataset(dataset_name, DATASET_SPLITS[dataset_name][0], debug_run)


def load_test_set(dataset_name: str, debug_run: bool):
    return _load_dataset(dataset_name, DATASET_SPLITS[dataset_name][1], debug_run)


def load_validation_set(dataset_name: str, debug_run: bool):
    return _load_dataset(dataset_name, DATASET_SPLITS[dataset_name][2], debug_run)


def _load_dataset(dataset_name: str, split: str, debug_run: bool) -> List[Dict]:
    if dataset_name in DATASETS:
        if split in DATASET_SPLITS[dataset_name]:
            mod_dataset_name = dataset_name.replace("/", "_")
            dataset_dir = debug_run_DIR if debug_run else FULL_RUN_DIR
            with open(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                    dataset_dir,
                    f"{mod_dataset_name}-{split}.json",
                ),
                "rb",
            ) as f:
                return json.load(f)
        else:
            sys.exit(f"Split, '{split}', not valid for dataset, '{dataset_name}'")
    else:
        sys.exit(
            f"Failed loading dataset, '{dataset_name}'. You can add it to the project"
        )

DATASETS = ["billsum", "cnn_dailymail", "reddit_tifu/long", "multi_news"]
DATASET_SPLITS = {
    "cnn_dailymail": ["train", "test", "validation"],
    "billsum": ["train", "test", "ca_test"],
    "reddit_tifu/long": ["train", "test"],
    "multi_news": ["train", "test", "validation"],
}

DATASET_KEYS = {
    "cnn_dailymail": {"input": "article", "output": "highlights", "meta": []},
    "reddit_tifu/long": {"input": "documents", "output": "tldr", "meta": []},
    "billsum": {"input": "text", "output": "summary", "meta": ["title"]},
    "multi_news": {"input": "document", "output": "summary", "meta": []},
}

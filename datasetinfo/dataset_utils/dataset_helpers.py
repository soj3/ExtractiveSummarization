from typing import List, Tuple
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

stemmer = PorterStemmer()
SYMBOLS = "#$%&()*+,-/:;<=>@[\]^_{|}~\"`"


def process_multi_news(text: str) -> List[Tuple[List[List[str]], List[List[str]]]]:
    split_texts = text.split("|||||")
    return [process_text(split_text) for split_text in split_texts if split_text != ""]


def process_text(text: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Run cleaning and tokenizing on text

    Args:
        text (str): input text to be processed

    Returns:
        Tuple[List[List[str]], List[List[str]]]: List of sentences of list of words and stemmed words
    """
    text = _clean_text(text)
    token_text = _tokenize_text(text)
    return token_text, _stem_clean_symbols(token_text)


def _stem_clean_symbols(token_text: List[List[str]]) -> List[List[str]]:
    return [
        [stemmer.stem(word.lower()) for word in sentence if word not in SYMBOLS]
        for sentence in token_text
    ]


def _clean_text(text: str) -> str:
    """Cleans up a given text by removing unused symbols

    Args:
        text (str): input text to be cleaned

    Returns:
        str: cleaned text
    """
    # remove newlines and cr
    # text = text.replace("\n", " ").replace("\r", " ")

    # # replacement list
    # repl_list = "#$%&()*+,-/:;<=>@[\]^_{|}~"
    # translator = str.maketrans(dict.fromkeys(repl_list, " "))
    # text = text.translate(translator)

    # replace single quotes
    translator = str.maketrans(dict.fromkeys("'", ""))
    text = text.translate(translator)

    return text


def _tokenize_text(text: str) -> List[List[str]]:
    """splits text into sentences and then into words

    Args:
        text (str): input text

    Returns:
        List[List[str]]: output of list of sentences of list of words
    """
    sentences = sent_tokenize(text)
    if len(sentences) == 1:
        return [word_tokenize(sentences[0])]
    return [word_tokenize(sent) for sent in sentences]

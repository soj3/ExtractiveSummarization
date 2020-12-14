from utils import generate_prob_dict


def generate_summary_basic(documents, summary_length):
    word_probs = generate_prob_dict(documents)
    summary_sentences = []
    for i in range(summary_length):
        idxs, word_probs = select_sentence(documents, word_probs)
        doc_idx, sent_idx = idxs
        summary_sentences.append((documents[doc_idx].original[sent_idx]))
    return summary_sentences


def weight_sentence(sentence, word_probs):
    weight = 0
    for word in sentence:
        weight += word_probs[word] / len(sentence)
    return weight


def select_sentence(documents, word_probs):
    max_weight = 0
    sentence_loc = None
    max_prob_word = max(word_probs, key=lambda key: word_probs[key])
    for i in range(len(documents)):
        document = documents[i]
        for j in range(len(document.text)):
            sentence = document.text[j]
            if max_prob_word in sentence:
                weight = weight_sentence(sentence, word_probs)
                if weight > max_weight:
                    max_weight = weight
                    sentence_loc = (i, j)
                    word_probs = update_word_probs(sentence, word_probs)

    return sentence_loc, word_probs


def update_word_probs(sentence, word_probs):
    for word in sentence:
        word_probs[word] = word_probs[word] ** 2
    return word_probs


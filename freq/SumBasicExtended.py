import math
from copy import deepcopy
from utils import generate_prob_dict, flatten


BIAS = 0.5
STACKSIZE = 20


def calc_pos_scores(documents):
    word_occurences = {}
    word_scores = {}

    for document in documents:
        words = flatten(document.text)
        for i in range(len(words)):
            if words[i] in word_occurences:
                word_occurences[words[i]].append((i) / (len(words) - 1))
            else:
                word_occurences[words[i]] = [(i) / (len(words) - 1)]

    for word in word_occurences:
        word_scores[word] = sum(word_occurences[word]) / len(word_occurences[word])
    return word_scores


def combine_scores(pos_scores, word_probs):
    new_scores = {}
    for word in word_probs:
        new_scores[word] = (
            BIAS * (1.01 - pos_scores[word]) + (1 - BIAS) * word_probs[word]
        )

    return new_scores


def generate_summary_extended(documents, summary_length):
    pos_scores = calc_pos_scores(documents)
    word_probs = generate_prob_dict(documents)
    combined_scores = combine_scores(pos_scores, word_probs)

    stack = [[], []]
    stack[0] = [(math.inf, [])]

    for i in range(summary_length):
        if i < len(stack):
            for tup in stack[i]:
                for j in range(len(documents)):
                    for k in range(len(documents[j].text)):
                        doc_id = j
                        sent_id = k
                        _, sol = deepcopy(tup)
                        if (doc_id, sent_id) in sol:
                            pass
                        else:
                            sol.append((doc_id, sent_id))
                            score = score_solution(sol, combined_scores, documents)
                            if i + 1 < len(stack) -1:
                                stack[i + 1].append((score, sol))
                            else:
                                stack.append([(score, sol)])
                            stack[i + 1].sort(reverse=True)
                            stack[i + 1] = stack[i + 1][:STACKSIZE]

    best_summary = max(stack[-1], key=lambda key: stack[-1][0])
    summary = []
    for idx in sorted(best_summary[1], key=lambda element: element[1]):
        doc_id, sent_id = idx
        summary.append(documents[doc_id].original[sent_id])

    return summary


def score_solution(solution, word_scores, documents):
    score = 0
    counted = []
    for (doc_id, sent_id) in solution:
        for word in documents[doc_id].text[sent_id]:
            if word not in counted:
                score += word_scores[word]
                counted.append(word)

    return score

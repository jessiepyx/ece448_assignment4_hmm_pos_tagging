"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math


def baseline(train, test):
    """
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    """

    cnt_tag = dict()
    cnt_word_tag = dict()
    for sentence in train:
        for (word, tag) in sentence:
            val = cnt_tag.get(tag, 0)
            cnt_tag[tag] = val + 1
            if word in cnt_word_tag:
                val = cnt_word_tag[word].get(tag, 0)
                cnt_word_tag[word][tag] = val + 1
            else:
                cnt_word_tag[word] = {tag: 1}
    word_tag = dict()
    for word, tags in cnt_word_tag.items():
        word_tag[word] = max(tags, key=lambda x: tags[x])
    most_frequent_tag = max(cnt_tag, key=lambda x: cnt_tag[x])
    predicts = []
    for sentence in test:
        res = []
        for word in sentence:
            res.append((word, word_tag.get(word, most_frequent_tag)))
        predicts.append(res)
    return predicts


def viterbi_p1(train, test):
    """
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    """

    laplace_smooth = 1e-5
    tags = set()
    cnt_tag = dict()
    cnt_tag_start = dict()
    cnt_tag_end = dict()
    cnt_tag_pair = dict()
    cnt_tag_word = dict()
    vocabulary = set()
    for sentence in train:
        for k in range(len(sentence)):
            word, tag = sentence[k]
            tags.add(tag)
            val = cnt_tag.get(tag, 0)
            cnt_tag[tag] = val + 1
            if k == 0:
                val = cnt_tag_start.get(tag, 0)
                cnt_tag_start[tag] = val + 1
            else:
                if k == len(sentence) - 1:
                    val = cnt_tag_end.get(tag, 0)
                    cnt_tag_end[tag] = val + 1
                prev_tag = sentence[k - 1][1]
                val = cnt_tag_pair.get((prev_tag, tag), 0)
                cnt_tag_pair[(prev_tag, tag)] = val + 1
            val = cnt_tag_word.get((tag, word), 0)
            cnt_tag_word[(tag, word)] = val + 1
            vocabulary.add(word)

    # Initial probability
    log_initial_p = dict()
    num_of_starting_position = sum(cnt_tag_start.values())
    for tag in tags:
        log_initial_p[tag] = math.log(
            (cnt_tag_start.get(tag, 0) + laplace_smooth) / (
                    num_of_starting_position + laplace_smooth * len(tags)))

    # Transition probability
    log_transition_p = dict()
    for tag_prev in tags:
        for tag_curr in tags:
            log_transition_p[(tag_prev, tag_curr)] = math.log(
                (cnt_tag_pair.get((tag_prev, tag_curr), 0) + laplace_smooth) / (
                        cnt_tag[tag_prev] - cnt_tag_end.get(tag_prev, 0) + laplace_smooth * len(tags)))

    # Emission probability
    log_emission_p = dict()
    for sentence in test:
        for word in sentence:
            vocabulary.add(word)
    for tag in tags:
        for word in vocabulary:
            log_emission_p[(tag, word)] = math.log(
                (cnt_tag_word.get((tag, word), 0) + laplace_smooth) / (
                        cnt_tag[tag] + laplace_smooth * len(vocabulary)))

    predicts = []
    for sentence in test:
        trellis = []
        nodes_edges = dict()
        path = dict()
        for k, word in enumerate(sentence):
            if k == 0:
                curr = dict()
                for tag_curr in tags:
                    curr[tag_curr] = log_initial_p[tag_curr] + log_emission_p[(tag_curr, word)]
                trellis.append(curr)
            else:
                prev = trellis[-1]
                curr = dict()
                for tag_curr in tags:
                    for tag_prev in tags:
                        nodes_edges[(tag_prev, tag_curr)] = prev[tag_prev] + log_transition_p[
                            (tag_prev, tag_curr)] + log_emission_p[(tag_curr, word)]
                    select_tag_prev = max(tags, key=lambda x: nodes_edges[(x, tag_curr)])
                    curr[tag_curr] = nodes_edges[(select_tag_prev, tag_curr)]
                    path[(k, tag_curr)] = select_tag_prev
                trellis.append(curr)
        tag = max(tags, key=lambda x: trellis[-1][x])
        res = [(sentence[-1], tag)]
        for k in range(len(sentence) - 1, 0, -1):
            tag = path[(k, tag)]
            res.insert(0, (sentence[k - 1], tag))
        predicts.append(res[:])

    return predicts


def viterbi_p2(train, test):
    """
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    """

    cnt_tag = dict()
    cnt_word_tag = dict()
    for sentence in train:
        for (word, tag) in sentence:
            val = cnt_tag.get(tag, 0)
            cnt_tag[tag] = val + 1
            if word in cnt_word_tag:
                val = cnt_word_tag[word].get(tag, 0)
                cnt_word_tag[word][tag] = val + 1
            else:
                cnt_word_tag[word] = {tag: 1}
    word_tag = dict()
    for word, tags in cnt_word_tag.items():
        word_tag[word] = max(tags, key=lambda x: tags[x])
    most_frequent_tag = max(cnt_tag, key=lambda x: cnt_tag[x])
    predicts = []
    for sentence in test:
        res = []
        for word in sentence:
            res.append((word, word_tag.get(word, most_frequent_tag)))
        predicts.append(res)
    return predicts

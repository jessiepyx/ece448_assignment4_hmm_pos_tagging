from collections import defaultdict
import math


def extra(train, test):
    '''
    TODO: implement improved viterbi algorithm for extra credits.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    laplace_smooth = 1e-5
    tags = set()
    cnt_tag = dict()
    cnt_tag_start = dict()
    cnt_tag_end = dict()
    cnt_tag_pair = dict()
    cnt_word_tag = dict()
    vocabulary = set()
    cnt_word = dict()
    cnt_tag_hapax = dict()
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
            if word in cnt_word_tag:
                val = cnt_word_tag[word].get(tag, 0)
                cnt_word_tag[word][tag] = val + 1
            else:
                cnt_word_tag[word] = {tag: 1}
            vocabulary.add(word)
            val = cnt_word.get(word, 0)
            cnt_word[word] = val + 1

    # Hapax probability
    hapax_p = dict()
    # dump = []
    for (word, times) in cnt_word.items():
        if times == 1:
            tag = list(cnt_word_tag[word].keys())[0]
            val = cnt_tag_hapax.get(tag, 0)
            cnt_tag_hapax[tag] = val + 1
            # dump.append(tag + ' ' + word)
    for tag in tags:
        hapax_p[tag] = (cnt_tag_hapax.get(tag, 0) + laplace_smooth) / (
                sum(cnt_tag_hapax.values()) + laplace_smooth * len(tags))
    # f = open('data/hapax_dump.txt', 'w+')
    # f.write('\n'.join(sorted(dump)))

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
    unseen = set()
    for sentence in test:
        for word in sentence:
            if word not in vocabulary:
                unseen.add(word)
    for tag in tags:
        total_occurrence_unseen = hapax_p[tag] * len(unseen)
        for word in vocabulary:
            log_emission_p[(tag, word)] = math.log(
                (cnt_word_tag[word].get(tag, 0) + laplace_smooth) / (
                        cnt_tag[tag] + laplace_smooth * (len(vocabulary) + total_occurrence_unseen)))
        for word in unseen:
            log_emission_p[(tag, word)] = math.log(
                laplace_smooth * hapax_p[tag] / (
                        cnt_tag[tag] + laplace_smooth * (len(vocabulary) + total_occurrence_unseen)))

    numbers = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
        'hundred', 'thousand', 'million', 'billion', 'trillion',
    ]

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
        for k in range(len(res)):
            word = res[k][0]
            if word in unseen:
                if '$' in word:
                    res[k] = (word, 'NOUN')
                elif len(word) > 2 and word[-2:] == "'s":
                    res[k] = (word, 'NOUN')
                elif len(word) > 3 and word[-3:] == 'ist':
                    res[k] = (word, 'NOUN')
                elif len(word) > 2 and word[-2:] == 'ty':
                    res[k] = (word, 'NOUN')
                elif len(word) > 3 and word[-3:] == 'ter':
                    res[k] = (word, 'NOUN')
                elif len(word) > 3 and word[-2:] == 'tor':
                    res[k] = (word, 'NOUN')
                elif len(word) > 4 and word[-3:] == 'ists':
                    res[k] = (word, 'NOUN')
                elif len(word) > 3 and word[-2:] == 'ties':
                    res[k] = (word, 'NOUN')
                elif len(word) > 4 and word[-4:] == 'ters':
                    res[k] = (word, 'NOUN')
                elif len(word) > 4 and word[-4:] == 'tors':
                    res[k] = (word, 'NOUN')
                elif len(word) > 3 and word[-3:] == 'ism':
                    res[k] = (word, 'NOUN')
                elif len(word) > 2 and word[-2:] == 'ly':
                    res[k] = (word, 'ADV')
                elif len(word) > 3 and word[-3:] == 'ble':
                    res[k] = (word, 'ADJ')
                elif len(word) > 3 and word[-3:] == 'ful':
                    res[k] = (word, 'ADJ')
                elif '-' in word and len(word) > 2 and word[-2:] == 'ed':
                    res[k] = (word, 'ADJ')
                elif '-' in word and len(word) > 3 and word[-3:] == 'ing':
                    res[k] = (word, 'ADJ')
                elif '-' in word and res[k][1] != 'ADJ':
                    res[k] = (word, 'NOUN')
                elif len(word) > 2 and word[-2:] == 'al':
                    res[k] = (word, 'ADJ')
                elif len(word) > 3 and word[-3:] == 'ing':
                    res[k] = (word, 'VERB')
                elif len(word) > 2 and word[-2:] == 'ed':
                    res[k] = (word, 'VERB')
                elif any(substring in numbers for substring in word):
                    res[k] = (word, 'NUM')
        predicts.append(res[:])

    return predicts

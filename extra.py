from collections import defaultdict
import math
def extra(train,test):
    '''
    TODO: implement improved viterbi algorithm for extra credits.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

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
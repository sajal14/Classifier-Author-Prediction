from nltk import word_tokenize
import pickle
import numpy as np
from collections import Counter
import math

def standardize(rawexcerpt):
    l = word_tokenize(rawexcerpt.lower().decode("utf-8"))
    for i,word in enumerate(l):
        l[i] = word.encode("utf-8")
    return l


def generate_table(file_path,feature_space_bigrams):
    f = open(file_path,"r").read()
    lines = f.split("\n")
    set_fs = set(feature_space_bigrams)
    final_array = np.array([0] * len(feature_space_bigrams))

    ind_dict = {}
    i = 0
    for pa in feature_space_bigrams:
        ind_dict[pa] = i;
        i = i+1;

    print ind_dict

    print set_fs

    for line in lines:
        row_arr = np.array([0]* len(feature_space_bigrams))
        stdList = standardize(line)
        pairs = zip(stdList, stdList[1:])
        count_pairs = dict(Counter(pairs))
        # print pairs

        for p in count_pairs.keys():
            if p in set_fs:
                ind = ind_dict[p]
                row_arr[ind] = count_pairs[p]

        final_array = np.vstack((final_array,row_arr))

    return final_array





# all_articles = open("train/project_articles_train_modified").read()
# stdList = standardize(all_articles)
# pairs = zip(stdList, stdList[1:])
# bigrams = Counter(pairs).most_common(2000)
#
#
# bigram_fs = [a[0] for a in bigrams]
# # print bigrams
# pickle.dump(bigram_fs, open("top_2k_bigram_freq.p", 'wb'))

""" End of bigram feature select"""

feature_space_words = pickle.load(open("top_2k_bigram_freq.p", 'rb'))
# all_articles = open("test/project_articles_test").read()

np_bigram = generate_table("test/project_articles_test",feature_space_words)

# print feature_space_words
pickle.dump(np_bigram, open("np_bigram_freq_test.p", 'wb'))





# len_pairs = len(pairs)
# bigrams_author = Counter(pairs).most_common(1)
# print bigrams_author


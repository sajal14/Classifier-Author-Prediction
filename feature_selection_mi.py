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



def get_word_probs(sample):
    total_words = len(sample)
    dic = dict(Counter(sample))
    for k in dic:
        dic[k] = float(dic[k])/total_words
    return dic




def generate_table(file_path,unigram_fs):
    f = open(file_path,"r").read()
    lines = f.split("\n")
    set_fs = set(unigram_fs)
    final_array = np.array([0] * len(unigram_fs))

    ind_dict = {}
    i = 0
    for pa in unigram_fs:
        ind_dict[pa] = i;
        i = i+1;

    print ind_dict

    for line in lines:
        row_arr = np.array([0]* len(unigram_fs))

        count_uni = dict(Counter(standardize(line)))
        # print pairs
        for p in count_uni.keys():
            if p in set_fs:
                ind = ind_dict[p]
                row_arr[ind] = count_uni[p]

        final_array = np.vstack((final_array,row_arr))

    return final_array



dict_mi = {}

# print corpus
#
# sample = open("authorArticles.txt").read()
# corpus = open("project_articles_train_modified").read()
# dict_prob_in_corpus = get_word_probs(standardize(corpus))
#
# dict_prob_in_sample = get_word_probs(standardize(sample))
#
#
# word_count = dict(Counter(standardize(corpus)))
#
# for k in dict_prob_in_sample:
#     if(word_count[k] >= 5):
#         dict_mi[k] = math.log(float(dict_prob_in_sample[k])/dict_prob_in_corpus[k])
#
#
# sort_tups = sorted(dict_mi.items(),key = lambda x : x[1], reverse=True)
#
# f = open("top_mi_10k_new","w")
# top_ten_k = sort_tups[0:10000]
#
# pickle.dump(sort_tups, open("unigram_mi.p", 'wb'))

# print top_ten_k

unigram_mi = pickle.load(open("unigram_mi.p", 'rb'))

print unigram_mi

array = generate_table("project_articles_test",unigram_mi)


pickle.dump(array,open("unigram_test_table.p","wb"))
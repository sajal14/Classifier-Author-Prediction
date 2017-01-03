from nltk import word_tokenize
import pickle
import numpy as np
from collections import Counter
import math

def get_word_probs(sample):#Getting list of tuples here
    total_words = len(sample)
    dic = dict(Counter(sample))
    for k in dic:
        dic[k] = float(dic[k])/total_words
    return dic



def generate_table(file_path,feature_space_trigrams):
    f = open(file_path,"r").read()
    lines = f.split("\n")
    set_fs = set(feature_space_trigrams)
    final_array = np.array([0] * len(feature_space_trigrams))

    ind_dict = {}
    i = 0
    for pa in feature_space_trigrams:
        ind_dict[pa] = i;
        i = i+1;

    print ind_dict

    print set_fs

    for line in lines:
        row_arr = np.array([0]* len(feature_space_trigrams))
        stdList = word_tokenize(line) #No decoding encoding required
        tries = zip(stdList, stdList[1:], stdList[2:])
        count_tries = dict(Counter(tries))
        # print pairs

        for p in count_tries.keys():
            if p in set_fs:
                ind = ind_dict[p]
                row_arr[ind] = count_tries[p]

        final_array = np.vstack((final_array,row_arr))

    return final_array



#
# f = open("train_pos.txt").read()
# # stdList = standardize(f)
#
#
# stdList = word_tokenize(f) #no decoding encoding required
# # print stdList
# all_trigrams = zip(stdList, stdList[1:], stdList[2:])
# # print all_trigrams
#
# dict_all = get_word_probs(all_trigrams)
#
# # print dict_all
#
#
#
# #Reading pos of only author articles:
#
# f = open("authorArticlesPOS.txt").read()
# stdList = word_tokenize(f) #no decoding encoding required
# author_trigrams = zip(stdList, stdList[1:], stdList[2:])
# dict_author = get_word_probs(author_trigrams)
#
# tri_count = dict(Counter(author_trigrams))
#
# dict_mi = {}
# for k in dict_author:
#     if tri_count[k] >=5:
#         if k in dict_all:
#             dict_mi[k] = math.log(float(dict_author[k])/dict_all[k])
#
#
# # print dict_mi
#
# sort_tups = sorted(dict_mi.items(),key = lambda x : x[1], reverse=True)
#
# # print sort_tups[0:27]
#
#
# top_tries = [a[0] for a in sort_tups]
# feature_space_words = top_tries[0:27]
# pickle.dump(feature_space_words, open("feature_space_pos_tri.p", 'wb'))


""" End of feature space creation code"""

feature_space_words = pickle.load(open("feature_space_pos_tri.p", 'rb'))


np_trigram_mi = generate_table("test_pos.txt",feature_space_words)

print np.nonzero(np_trigram_mi)

pickle.dump(np_trigram_mi,open("test_pos_tri.p","wb"))

















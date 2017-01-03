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



dict_bigrams_sample ={}

dict_bigrams_corpus = {}

def get_word_probs(sample):#Getting list of tuples here
    total_words = len(sample)
    dic = dict(Counter(sample))
    for k in dic:
        dic[k] = float(dic[k])/total_words
    return dic


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

# f = open("biGram10000.txt").read()
#
# all_bigrams = f.split("\n")
#
# all_bigrams = [x.strip('\r') for x in all_bigrams]
#
# # print all_bigrams
#
# author_articles = open("authorArticles.txt").read()
# stdList = standardize(author_articles)
# pairs = zip(stdList, stdList[1:])
# # len_pairs = len(pairs)
# # bigrams_author = Counter(pairs).most_common(1)
# # print bigrams_author
#
# dict_author = get_word_probs(pairs)
# # print dict_author
# pair_count = dict(Counter(pairs))
#
#
# all_articles = open("project_articles_train_modified").read()
# stdList = standardize(all_articles)
# pairs = zip(stdList, stdList[1:])
# # len_pairs = len(pairs)
# # bigrams_author = Counter(pairs).most_common(1)
# # print bigrams_author
#
# dict_all = get_word_probs(pairs)
#
# print dict_author[('analysis','gastroenterologists')]
#
#
#
# dict_mi = {}
# for k in dict_author:
#     if(pair_count[k] >= 5):
#         print k
#         if k in dict_all:
#             dict_mi[k] = math.log(float(dict_author[k])/dict_all[k])
#
#
# sort_tups = sorted(dict_mi.items(),key = lambda x : x[1], reverse=True)
#
#
# # print sort_tups[0:1000]
# # print len(sort_tups)
#
# top_pairs = [a[0] for a in sort_tups]
# feature_space_words = top_pairs[0:672]
# pickle.dump(feature_space_words, open("feature_space_mi.p", 'wb'))
# print feature_space_words



feature_space_words = pickle.load(open("feature_space_mi.p", 'rb'))

# print feature_space_words

# np_bigram_mi = generate_table("project_articles_train_modified",feature_space_words)

np_bigram_mi = generate_table("project_articles_test",feature_space_words)

print np_bigram_mi

print np.nonzero(np_bigram_mi)


# pickle.dump(np_bigram_mi,open("np_from_mi.p","wb"))
pickle.dump(np_bigram_mi,open("test_data_mi_new.p","wb"))

mi_test_data = pickle.load(open("test_data_mi_new.p", 'rb'))

print len(mi_test_data)
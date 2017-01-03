import os
import re
import numpy as np
from collections import Counter
# from svmutil import *
from sklearn.preprocessing import normalize
import math
import ast
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")
from sklearn import svm


def train_test_model(train_datafile, test_datafile):
    y,x  = svm_read_problem(train_datafile)
    y_test,x_test = svm_read_problem(test_datafile)
    m = svm_train(y,x,'-t 0 -e .01 -m 1000 -h 0')
    # print y_test,x_test
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    return p_label, p_acc, p_val


def standardize(rawexcerpt):
    l = word_tokenize(rawexcerpt.lower().decode("utf-8"))
    for i,word in enumerate(l):
        l[i] = word.encode("utf-8")
    return l


def split_data(list_of_tups):
    length = len(list_of_tups)
    tr_size = int(0.8*length)
    tr_data = list_of_tups[0:tr_size]
    test_data = list_of_tups[tr_size:length]
    return tr_data,test_data


def create_feature_vector(file_path,sorted_vocab_list,sorted_dict,l,u): #lower index to upper index
    r2 = open(file_path,"r").read()
    r2 = r2.split("\n")
    # l= len(sorted_vocab_list)
    #l = k +1  #First K array #+1 for <UNK>
    # l = k
    len1 = (u - l) + 1
    final_array = np.array([0] * len1)
    for line in r2:
        row_arr = np.array([0]* len1)
        cntr = Counter(standardize(line)).most_common()
        for tups in cntr:
            word = tups[0]
            # print word
            if word == "n.y":
                word = word + "."
            if word == "d.c":
                word = word + "."
            if word == "u.s":
                word = word + "."
            if word == "n.c":
                word = word + "."
            #
            # if len(word)>1 and word.endswith(":"):
            #     word = word[0:len(word)-1]
            #     row_arr[sorted_dict[":"]] += 1
            if word == "cancer:":
                word = "cancer"

            if word == "minn":
                word = "minn."

            if word == "2015":
                word = "2015."



            if word not in sorted_dict.keys():
                 if word.endswith("."):
                     nw = word[0:len(word)-1]
                     word = nw
                 else:
                     word += "."

            if word not in sorted_dict:
                continue

            ind = sorted_dict[word]

            if ind > u or ind < l: #UNK handling
                continue #Dont put it
            # print ind
            row_arr[ind-l] = tups[1]
        final_array = np.vstack((final_array,row_arr))


    # np.delete(final_array,0,0)
    return final_array

# def train_model(np_array,y):
#     clf = svm.SVC(kernel = "linear", C=1)
#     scores = cross_validation.cross_val_score(clf,np_arr,y)


if __name__ == "__main__":
    # r = open("project_articles_train_modified","r").read()
    # print(type(r))

    #Making Vocab Dict
    # words = standardize(r)
    # # print words
    # # counter = Counter(words)
    # l = list(Counter(words).most_common())
    #
    # ind = 0;
    # vocab_dict ={}
    # sorted_vocab_list = []
    # for w in l:
    #     vocab_dict[w[0]] = ind
    #     ind = ind+1
    #     sorted_vocab_list.append(w[0])
    #
    # pickle.dump(vocab_dict, open("vocab_dict.p", 'wb'))
    # pickle.dump(sorted_vocab_list, open("sorted_vocab_list.p", 'wb'))


    vocab_dict = pickle.load(open("vocab_dict.p", 'rb'))
    sorted_vocab_list = pickle.load(open("sorted_vocab_list.p",'rb'))
    # all_feature = create_feature_vector("project_articles_train_modified",sorted_vocab_list,vocab_dict, 4000,4999)
    all_feature = create_feature_vector("project_articles_test",sorted_vocab_list,vocab_dict, 0,1999)

    # project_articles_train_modified",sorted_vocab_list,vocab_dict,5000)
    # print all_feature

    np_arr = pickle.dump(all_feature, open("test_top_2k.p", 'wb'))


    # k = 2000
    # col_len = np_arr[0].size
    #
    # new_np_arr = np_arr[:,range(0,k)]
    # rest = np_arr[k,col_len]
    # rest = rest.sum(1)
    # new_np_arr = np.hstack(new_np_arr,rest)

    # np.delete(np_arr,range(2000,col_len),1)


    # classifier_model = train_model(new_np_arr)


    # vocab = {}
    # vocab_set = set()
    # for line in r:
    #     words = word_tokenize(line.de)
    #     vocab_set.union(set(words))

        # vocab_set.add(words)





import nltk

from nltk import word_tokenize

def standardize(rawexcerpt):
    l = word_tokenize(rawexcerpt.lower().decode("utf-8"))
    for i,word in enumerate(l):
        l[i] = word.encode("utf-8")
    return l


f = open("test/project_articles_test","r").read()


lines = f.split("\n")

print len(lines)


file = open("test_pos.txt","w")

for line in lines:
    tups =  nltk.pos_tag(word_tokenize(line.lower().decode("utf-8")))
    # print tups
    poss =  [y for (x,y) in tups]
    file.write(" ".join(poss))
    file.write("\n")
    # print poss
    # break
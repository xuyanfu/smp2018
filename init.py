import pickle
import jieba
import re
from tqdm import tqdm
import numpy as np



def init_train():
    with open('data/xtrain.pkl','rb') as input:
        xtrain = pickle.load(input)

    with open('data/ytrain.pkl','rb') as input:
        ytrain = pickle.load(input)

    with open('data/xtest.pkl','rb') as input:
        xtest = pickle.load(input)

    with open('data/ytest.pkl','rb') as input:
        ytest = pickle.load(input)

    print (123)

    print (len(xtrain),len(ytrain),len(xtest),len(ytest))

    set_len = set()
    list_len = []
    fixlen = 350

    '''
    for x in xtrain:
        tmp_sentence = re.split('。|？|！',x)
        set_len.add(len(tmp_sentence))
        list_len.append(len(tmp_sentence))
        for sentence in tmp_sentence:
            words = list (jieba.cut(sentence))
            print (words)
    avg_len = sum(list_len) / len(list_len)
    print (123)
    '''

    '''
    for x in tqdm(xtrain):
        words = list(jieba.cut(x))
        list_len.append(len(words))
    
    avg_len = sum(list_len) / len(list_len)
    print (avg_len)
    print (123)
    '''

    xtrain_words = []
    xtest_words = []

    for x in tqdm(xtrain):
        xtrain_words.append(list(jieba.cut(x)))

    for x in tqdm(xtest):
        xtest_words.append(list(jieba.cut(x)))


    dict_word2id = {}
    
    for x in tqdm(xtrain_words):
        #words = list (jieba.cut(x))
        words = x[:fixlen]
        for word in words:
            if word not in dict_word2id.keys():
                dict_word2id[word] = len(dict_word2id)
            #break
        #break
        #print (words)
    print (len(dict_word2id))
    
    for x in tqdm (xtest_words):
        #words = list (jieba.cut(x))
        words = x[:fixlen]
        for word in words:
            if word not in dict_word2id.keys():
                dict_word2id[word] = len(dict_word2id)
            #break
        #break
        #print (words)
    print (len(dict_word2id))
    
    with open('data/dict_word2id.pkl','wb') as output:
        pickle.dump(dict_word2id,output)


    with open('data/dict_word2id.pkl','rb') as input:
        dict_word2id = pickle.load(input)

    dict_word2id['UNK'] = len(dict_word2id)
    dict_word2id['BLANK'] = len(dict_word2id)
    print (len(dict_word2id))

    list_xtrain = []#[[wordid*fixlen],[wordid*fixlen] ... ]
    list_xtest = []#[[wordid*fixlen],[wordid*fixlen] ... ]
    for x in tqdm(xtrain_words):
        #words = list (jieba.cut(x))
        words = x[:fixlen]

        tmp_train = []
        for i in range(fixlen):
            wordid = dict_word2id['BLANK']
            tmp_train.append(wordid)

        for index in range(len(words)):
            if words[index] not in dict_word2id.keys():
                wordid = dict_word2id['UNK']
            else:
                wordid = dict_word2id[words[index]]
            tmp_train[index] = wordid

        list_xtrain.append(tmp_train)
            #break
        #break
        #print (words)

    for x in tqdm(xtest_words):
        #words = list (jieba.cut(x))
        words = x[:fixlen]

        tmp_test = []
        for i in range(fixlen):
            wordid = dict_word2id['BLANK']
            tmp_test.append(wordid)

        for index in range(len(words)):
            if words[index] not in dict_word2id.keys():
                wordid = dict_word2id['UNK']
            else:
                wordid = dict_word2id[words[index]]
            tmp_test[index] = wordid

        list_xtest.append(tmp_test)

    list_xtest = np.array(list_xtest)
    list_xtrain = np.array(list_xtrain)

    np.save('data/list_xtrain.npy',list_xtrain)
    np.save('data/list_xtest.npy',list_xtest)

    list_ytrain = [] #[[4*one hot],[4*one hot],[4*one hot]]
    list_ytest =  [] #[[4*one hot],[4*one hot],[4*one hot]]

    for y in ytrain:
        tmp_label = [0 for _ in range(4)]
        tmp_label[y] = 1
        list_ytrain.append(tmp_label)
        #break

    for y in ytest:
        tmp_label = [0 for _ in range(4)]
        tmp_label[y] = 1
        list_ytest.append(tmp_label)
        #break

    list_ytrain = np.array(list_ytrain)
    list_ytest  = np.array(list_ytest)
    np.save('data/list_ytrain.npy',list_ytrain)
    np.save('data/list_ytest.npy',list_ytest)

    print (123)




def init_validation():

    with open('data/x_validation.pkl','rb') as input:
        x_validation = pickle.load(input)

    with open('data/dict_word2id.pkl', 'rb') as input:
        dict_word2id = pickle.load(input)

    dict_word2id['UNK'] = len(dict_word2id)
    dict_word2id['BLANK'] = len(dict_word2id)
    print('dict_word2id:',len(dict_word2id))

    fixlen = 350

    x_validation_words = []

    for x in tqdm(x_validation):
        x_validation_words.append(list(jieba.cut(x)))

    list_x_validation = []  #[[wordid*fixlen],[wordid*fixlen] ... ]
    for x in tqdm(x_validation_words):
        # words = list (jieba.cut(x))
        words = x[:fixlen]

        tmp_train = []
        for i in range(fixlen):
            wordid = dict_word2id['BLANK']
            tmp_train.append(wordid)

        for index in range(len(words)):
            if words[index] not in dict_word2id.keys():
                wordid = dict_word2id['UNK']
            else:
                wordid = dict_word2id[words[index]]
            tmp_train[index] = wordid

        list_x_validation.append(tmp_train)
        # break
        # break
        # print (words)
    list_x_validation = np.array(list_x_validation)
    np.save('data/list_x_validation.npy', list_x_validation)
    print ('list_x_validation:',len(list_x_validation))

def init_lengths():
    list_xtrain = np.load('data/list_xtrain.npy')
    list_xtest = np.load('data/list_xtest.npy')
    list_x_validation = np.load('data/list_x_validation.npy')

    list_xtrain_len = []
    list_xtest_len = []
    list_x_validation_len = []


    for i in list_xtrain:
        i = list(i)

        if 435614 in i:
            tmp_len = i.index(435614) + 1
        else:
            tmp_len=350

        list_xtrain_len.append(tmp_len)

    for i in list_xtest:
        i = list(i)

        if 435614 in i:
            tmp_len = i.index(435614) + 1
        else:
            tmp_len=350

        list_xtest_len.append(tmp_len)

    for i in list_x_validation:
        i = list(i)

        if 435614 in i:
            tmp_len = i.index(435614) + 1
        else:
            tmp_len=350

        list_x_validation_len.append(tmp_len)

    list_xtrain_len = np.array(list_xtrain_len)
    list_xtest_len = np.array(list_xtest_len)
    list_x_validation_len = np.array(list_x_validation_len)

    np.save('data/list_xtrain_len.npy',list_xtrain_len)
    np.save('data/list_xtest_len.npy', list_xtest_len)
    np.save('data/list_x_validation_len.npy', list_x_validation_len)


#init_validation()

init_lengths()
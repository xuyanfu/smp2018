import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import pickle



def test(save_path):
    pred = np.load(save_path)
    with open('data/ytest.pkl','rb') as input:
        ytest = pickle.load(input)

    pred =np.array(pred)
    ytest = np.array(ytest)
    result = np.equal(pred,ytest)
    #print (result)
    result = list(result)
    positive = result.count(True)
    print (positive)
    print (positive/len(ytest))

def validation():
    pred = np.load('data/validation_labels_epoch3.npy',)
    with open('data/id_validation.pkl','rb') as input:
        id_validationn = pickle.load(input)

    assert (len(pred) == len(id_validationn))
    length = len(pred)
    with open('result/origin_rnn_cnn_total_c300_r300_3.csv','w',encoding='utf-8') as output:
        for i in range(length):
            tmp_line = str (id_validationn[i]) + ',' + pred[i]+'\n'
            output.write(tmp_line)

#test('data/pred.npy')
#validation()
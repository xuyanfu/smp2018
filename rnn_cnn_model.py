import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import pickle
import test_pred


class Settings(object):
    def __init__(self):
        self.vocab_size = 435615
        self.len_sentence = 350
        self.num_epochs = 1
        self.num_classes = 4
        self.cnn_size = 300
        self.num_layers = 1
        self.word_embedding = 50
        self.keep_prob = 0.5
        self.batch_size = 300
        self.num_steps = 10000
        self.lr= 0.001
        self.gru_size = 300


class RNN_CNN():
    def __init__(self, setting):
        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes = setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr
        self.gru_size = setting.gru_size

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)
        self.input_length = tf.placeholder(tf.int32, [None])


        word_embedding = tf.get_variable('word_embedding',[self.vocab_size, self.word_embedding])

        self.input_data = tf.nn.embedding_lookup(word_embedding, self.input_word)



        #双向GRU层
        gru_fw_cell = tf.nn.rnn_cell.GRUCell(self.gru_size)
        gru_bw_cell = tf.nn.rnn_cell.GRUCell(self.gru_size)

        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                                         cell_bw=gru_bw_cell,
                                                                         inputs=self.input_data,
                                                                         sequence_length=self.input_length,
                                                                         dtype=tf.float32,
                                                                         scope="BiGRU")

        self.rnn_outputs = tf.concat([output_fw, output_bw], axis=2)


        cnn_input = tf.reshape(self.rnn_outputs, [-1,self.len_sentence,self.gru_size*2,1] )

        # 卷积层
        conv = layers.conv2d(inputs=cnn_input, num_outputs=self.cnn_size, kernel_size=[3, self.gru_size*2],
                             stride=[1, self.gru_size*2], padding='SAME')

        # pooling层
        max_pool = layers.max_pool2d(conv, kernel_size=[self.len_sentence, 1], stride=[1, 1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

        # dropout层
        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh, keep_prob=self.keep_prob)

        # 全连接层
        self.outputs = layers.fully_connected(inputs=drop, num_outputs=self.num_classes, activation_fn=tf.nn.softmax)

        # loss
        self.cross_loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1)))
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.final_loss = self.cross_loss + self.l2_loss

        # accuracy
        self.pred = tf.argmax(self.outputs, axis=1)
        self.pred_prob = tf.reduce_max(self.outputs, axis=1)

        self.y_label = tf.argmax(self.input_y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y_label), 'float'))

        # minimize loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.final_loss)

def train (save_path,cnn_num,rnn_num,i):

    print('reading training data')

    list_xtrain = np.load('data/list_xtrain.npy')
    list_xtest = np.load('data/list_xtest.npy')
    list_ytrain = np.load('data/list_ytrain.npy')
    list_ytest = np.load('data/list_ytest.npy')
    list_xtrain_len = np.load('data/list_xtrain_len.npy')
    list_xtest_len = np.load('data/list_xtest_len.npy')

    assert(len(list_ytrain) == len(list_xtrain) and len(list_xtrain) == len(list_xtrain_len) )


    list_xtrain = list(list_xtrain)
    list_xtest = list(list_xtest)
    list_ytrain = list(list_ytrain)
    list_ytest = list(list_ytest)
    list_xtrain += list_xtest
    list_ytrain += list_ytest
    assert (len(list_ytrain) == len(list_xtrain))


    print (len(list_xtrain),len(list_ytrain))

    settings = Settings()
    settings.num_classes = len(list_ytrain[0])
    settings.num_steps = (len(list_xtrain) // settings.batch_size) +1
    settings.cnn_size = cnn_num
    settings.gru_size = rnn_num

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = RNN_CNN(setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if i !=0:
                saver.restore(sess, save_path=save_path)

            for epoch in range(1, settings.num_epochs + 1):

                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))
                for _ in bar:
                #for _ in range(settings.num_steps):

                    sample_list = random.sample(range(len(list_ytrain)), settings.batch_size)
                    batch_train_word = [list_xtrain[x] for x in sample_list]
                    batch_train_y = [list_ytrain[x] for x in sample_list]
                    batch_train_len = [list_xtrain_len[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob
                    feed_dict[model.input_length] = batch_train_len
                    '''
                    output  = sess.run(model.rnn_outputs,feed_dict=feed_dict)
                    print (output.shape)
                    '''
                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    #break
                saver.save(sess, save_path=save_path)
                #break




def test (save_path,cnn_num,rnn_num):

    result = []#[labels]
    list_xtest = np.load('data/list_xtest.npy')
    list_ytest = np.load('data/list_ytest.npy')
    list_xtest_len = np.load('data/list_xtest_len.npy')

    assert (len(list_ytest) == len(list_xtest))

    settings = Settings()
    settings.num_classes = len(list_ytest[0])
    settings.num_steps = (len(list_xtest) // settings.batch_size) + 1
    settings.cnn_size = cnn_num
    settings.gru_size = rnn_num

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope("model"):
                model = RNN_CNN(setting=settings)

            saver = tf.train.Saver()

            saver.restore(sess, save_path=save_path)

            #for i in range(settings.num_steps + 1):
            for i in tqdm(range(settings.num_steps + 1)):

                batch_test_word = list_xtest[settings.batch_size * i: settings.batch_size * (i + 1)]
                batch_test_y = list_ytest[settings.batch_size * i: settings.batch_size * (i + 1)]
                batch_train_len = list_xtest_len[settings.batch_size * i: settings.batch_size * (i + 1)]

                feed_dict = {}
                feed_dict[model.input_word] = batch_test_word
                feed_dict[model.input_y] = batch_test_y
                feed_dict[model.keep_prob] = 1
                feed_dict[model.input_length] = batch_train_len

                pred = sess.run([model.pred],feed_dict=feed_dict)
                pred = list(pred[0])
                result += pred
    return result


def validation(save_path):

    result = []#[labels]

    list_xtest = np.load('data/list_x_validation.npy')
    list_xtest_len = np.load('data/list_x_validation_len.npy')
    dict_index = {0:'自动摘要', 1:'机器翻译',2:'人类作者',3:'机器作者'}
    #list_ytest = np.load('data/list_ytest.npy')
    assert (len(list_xtest) == len(list_xtest_len))

    settings = Settings()
    settings.num_classes = 4
    settings.num_steps = (len(list_xtest) // settings.batch_size) + 1

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope("model"):
                model = RNN_CNN(setting=settings)

            saver = tf.train.Saver()
            saver.restore(sess, save_path=save_path)

            for i in tqdm(range(settings.num_steps + 1)):

                batch_test_word = list_xtest[settings.batch_size * i: settings.batch_size * (i + 1)]
                batch_train_len = list_xtest_len[settings.batch_size * i: settings.batch_size * (i + 1)]
                #batch_test_y = [0 for _ in range(settings.batch_size)]

                feed_dict = {}
                feed_dict[model.input_word] = batch_test_word
                feed_dict[model.input_length] = batch_train_len
                #feed_dict[model.input_y] = batch_test_y
                feed_dict[model.keep_prob] = 1
                pred = sess.run([model.pred],feed_dict=feed_dict)
                pred = list(pred[0])
                result += pred

    assert (len(result) == len(list_xtest))
    validation_labels = [dict_index[x] for x in result ]
    return validation_labels

rnn_num = 200
cnn_num = 300

for i in range (3):
    print('rnn_num,cnn_num,i:', rnn_num, cnn_num, i)
    save_path = 'model/origin_rnncnn_model_total_c'+str(cnn_num)+'_r'+str(rnn_num)+'.ckpt'
    train(save_path,cnn_num,rnn_num,i)
    pred = test (save_path,cnn_num,rnn_num)
    np.save('data/origin_rnncnn_pred.npy',pred)
    test_pred.test(save_path ='data/origin_rnncnn_pred.npy')
    print ('\n')


#validation_labels = validation(save_path)
#np.save('data/validation_labels.npy',validation_labels)
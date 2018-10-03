# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:27:57 2018

@author: Me
"""

import tensorflow as tf
import data_providers as data_providers
import numpy as np
import time

batchSize = 50
train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
# print(train_data.next())
def getTrainBatch():
	# inputs, targets = train_data.next()
    return train_data.next()
def getValBatch():
    return valid_data.next()
def processBatch(batch):
    maxLength = 0
    batchLength = np.zeros([batchSize])
    for i in range(batchSize):
        batchLength[i] = len(batch[i])
        if maxLength < len(batch[i]):
            maxLength = len(batch[i])
    newBatch = np.zeros((batchSize,maxLength))
    for i in range(batchSize):
        newBatch[i] = np.pad(batch[i],(0,maxLength-len(batch[i])),'constant')
    return newBatch, batchLength
            
lstmUnits = 64
numClasses = 2
max_epoch = 10
maxSeqLength = 2505
numDimensions = 300 #Dimensions for each word vector
vocab_size = 93929

tf.reset_default_graph()

input_data = tf.placeholder(tf.int32, [batchSize,None]) #(b,L)
labels = tf.placeholder(tf.float32, [batchSize, numClasses]) #(b,2)
bacthLength = tf.placeholder(tf.int32, [batchSize])

embedding_table = tf.Variable(tf.random_normal([vocab_size, numDimensions]))

data = tf.nn.embedding_lookup(embedding_table, input_data) # data(b,max,d)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)

outputs, final_states = tf.nn.dynamic_rnn(lstmCell, data, bacthLength, dtype=tf.float32) #final_states.h (b,lstmUnits)

weight = tf.Variable(tf.random_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.zeros([numClasses]))

prediction = tf.matmul(final_states.h, weight) + bias

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = prediction))
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    step = 0
    sess.run(init)
    while step<max_epoch:
        step = step+1
        print("epoch:{}".format(step))
        train_data.new_epoch()
        iterations = 0
        while iterations<train_data.num_batches:
            iterations = iterations + 1
            print("epoch:{}".format(step),end=" ")
            print("iter:{}".format(iterations))
            train_batch = getTrainBatch()
            batch_data, batch_len = processBatch(train_batch[0])
            batch_label = train_batch[1]
            dict_feed = {}
            dict_feed[input_data] = batch_data
            dict_feed[labels] = batch_label
            dict_feed[bacthLength] = batch_len
            sess.run(optimizer, dict_feed)
            print("Acc:{}".format(sess.run(accuracy,dict_feed)))

        val_iter = 0
        val_loss = 0
        val_acc = 0
        valid_data.new_epoch()
        while val_iter<valid_data.num_batches:
            val_iter = val_iter + 1
            val_batch = getValBatch()
            val_batch_data, val_batch_len = processBatch(val_batch[0])
            val_batch_label = val_batch[1]
            dict_feed = {}
            dict_feed[input_data] = val_batch_data
            dict_feed[labels] = val_batch_label
            dict_feed[bacthLength] = val_batch_len
            val_loss = val_loss + sess.run(loss, dict_feed)
            val_acc = val_acc + sess.run(accuracy, dict_feed)
        print("val_loss:{}".format(val_loss/valid_data.num_batches))
        print("val_acc:{}".format(val_acc/valid_data.num_batches))
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

def luong_attention_layer(inputs, batchLength):
    # implement Effective Approaches to Attention-based Neural Machine Translation: https://arxiv.org/pdf/1508.04025.pdf
    numUnits = int(inputs.shape[2])
    W_a = tf.Variable(tf.random_normal([numUnits, numUnits], stddev = 0.1))
    score = tf.matmul(tf.expand_dims(tf.matmul(final_hidden_states, W_a), 1), inputs, transpose_b=True)
    score = tf.squeeze(score,[1])
    score_mask = tf.sequence_mask(batchLength, maxlen=tf.shape(score)[1])
    score_mask_values = float("-inf") * tf.ones_like(score)
    score = tf.where(score_mask, score, score_mask_values)
    alignments = tf.nn.softmax(score)
    context_vector = tf.matmul(tf.expand_dims(alignments, 1), inputs)
    context_vector = tf.squeeze(context_vector, [1])
    context_weight = tf.Variable(tf.random_normal([numUnits * 2, numUnits], stddev = 0.1))
    attention_vector = tf.tanh(tf.matmul(tf.concat([context_vector, final_hidden_states],1), context_weight))
    return attention_vector
    

def hierarchical_attention_layer(inputs, attention_units, batchLength):
    # implement Hierarchical Attention Networks for Document Classification: http://www.aclweb.org/anthology/N16-1174
    batchSize = int(inputs.shape[0])
    numUnits = int(inputs.shape[2])
    W_w = tf.Variable(tf.random_normal([numUnits, attention_units], stddev = 0.1))
    b_w = tf.Variable(tf.random_normal([attention_units], stddev = 0.1))
    u_w = tf.Variable(tf.random_normal([attention_units], stddev = 0.1))
    
    u_it = tf.tanh(tf.matmul(tf.reshape(inputs,[-1, numUnits]), W_w) + b_w) #(b * ?,a)
    score = tf.reshape(tf.matmul(u_it, tf.reshape(u_w,[attention_units,1])), [batchSize, -1]) #(b,?)
    score_mask = tf.sequence_mask(batchLength, maxlen=tf.shape(score)[1])
    score_mask_values = float("-inf") * tf.ones_like(score)
    score = tf.where(score_mask, score, score_mask_values)
    alignments = tf.nn.softmax(score)
    
    attention_vector = tf.reshape(tf.matmul(tf.reshape(alignments, [batchSize, 1, -1]), outputs), [batchSize, numUnits])
    return attention_vector
    
    

numUnits = 64
numClasses = 2
max_epoch = 20
maxSeqLength = 2505
embedding_dim = 100 #Dimensions for each word vector
vocab_size = 93929
num_layers = 1
dropout_keep_ratio = 1
bidirectional = True
attention = True
cell_method = 1
attention_size = 64
cnn = True
filter_size = 5
pool_size = 2

tf.reset_default_graph()

input_data = tf.placeholder(tf.int32, [batchSize,None]) #(b,?)
labels = tf.placeholder(tf.float32, [batchSize, numClasses]) #(b,2)
batchLength = tf.placeholder(tf.int32, [batchSize])

with tf.name_scope("word_embedding"):
    embedding_table = tf.Variable(tf.random_normal([vocab_size, embedding_dim], stddev = 0.1))

    data = tf.nn.embedding_lookup(embedding_table, input_data) # data(b,?,d)

with tf.name_scope("CNN"):
    if cnn:
        filter_weight = tf.Variable(tf.random_normal([filter_size, embedding_dim, 1, 1], stddev = 0.1))
        conv_b = tf.Variable(tf.constant(0.1, shape=[1]))
        conv_data = tf.nn.conv2d(tf.reshape(data,[batchSize, -1, embedding_dim, 1]), filter_weight, strides=[1,1,1,1], padding='VALID') #(b, L-f+1, d, 1)
        h = tf.nn.relu(tf.nn.bias_add(conv_data, conv_b))
        pool_data = tf.nn.max_pool(h, ksize = [1, pool_size, 1, 1], strides = [1,1,1,1],  padding='VALID')
        data = tf.reshape(pool_data, [batchSize, -1 ,1])
        max_len = np.argmax(batchLength)
        batchLength = tf.convert_to_tensor(np.ones([batchSize]) * max_len/2, dtype = tf.int32)
    
with tf.name_scope("RNN"):
    cell_methods = [tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.GRUCell]
    if not bidirectional:
        multiCells = [tf.nn.rnn_cell.DropoutWrapper(cell_methods[cell_method](numUnits), output_keep_prob = dropout_keep_ratio) for _ in range(num_layers)]

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(multiCells)

        outputs, final_states = tf.nn.dynamic_rnn(rnn_cells, data, batchLength, dtype=tf.float32) #final_states.h (b,numUnits)
        if cell_method == 0:
            final_hidden_states = final_states[-1].h #取最后一层LSTM的结果
        else:
            final_hidden_states = final_states[-1]
    else:
        multiCells_fw = [tf.nn.rnn_cell.DropoutWrapper(cell_methods[cell_method](numUnits), output_keep_prob = dropout_keep_ratio) for _ in range(num_layers)]
        multiCells_bw = [tf.nn.rnn_cell.DropoutWrapper(cell_methods[cell_method](numUnits), output_keep_prob = dropout_keep_ratio) for _ in range(num_layers)]
        
        rnn_cells_fw = tf.nn.rnn_cell.MultiRNNCell(multiCells_fw)
        rnn_cells_bw = tf.nn.rnn_cell.MultiRNNCell(multiCells_bw)
        
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(rnn_cells_fw, rnn_cells_bw, data, batchLength, dtype=tf.float32)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
        if cell_method == 0:
            final_hidden_states = tf.concat([final_state_fw[-1].h, final_state_bw[-1].h], 1)
        else:
            final_hidden_states = tf.concat([final_state_fw[-1], final_state_bw[-1]], 1)
        
        
        
with tf.name_scope("attention"):
    if attention:
        attention_vector = luong_attention_layer(outputs, batchLength)
        
with tf.name_scope("prediction"):
    if not bidirectional:
        weight = tf.Variable(tf.random_normal([numUnits, numClasses], stddev = 0.1))
    else:
        weight = tf.Variable(tf.random_normal([2 * numUnits, numClasses], stddev = 0.1))
    
    bias = tf.Variable(tf.random_normal([numClasses], stddev = 0.1))
    
    if not attention:
        prediction = tf.matmul(final_hidden_states, weight) + bias
    else:
        prediction = tf.matmul(attention_vector, weight) + bias
            
        
with tf.name_scope("acc"):
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

init = tf.global_variables_initializer()

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    
    train_writer = tf.summary.FileWriter(r"C:\Users\Me\Downloads\mlp project\trainSummary",sess.graph)
    valid_writer = tf.summary.FileWriter(r"C:\Users\Me\Downloads\mlp project\validSummary",sess.graph)
    step = 0
    sess.run(init)
    while step<max_epoch*train_data.num_batches:
        print("epoch:{}".format(step))
        train_data.new_epoch()
        iterations = 0
        while iterations<train_data.num_batches:
            step = step+1
            iterations = iterations + 1
            print("epoch:{}".format(step//train_data.num_batches + 1),end=" ")
            print("iter:{}".format(iterations))
            train_batch = getTrainBatch()
            batch_data, batch_len = processBatch(train_batch[0])
            batch_label = train_batch[1]
            dict_feed = {}
            dict_feed[input_data] = batch_data
            dict_feed[labels] = batch_label
            dict_feed[batchLength] = batch_len
            sess.run(optimizer, dict_feed)
            print("Acc:{}".format(sess.run(accuracy,dict_feed)))
            if iterations % 100 == 0:
                print("step:{}".format(step))
                train_summary = sess.run(merge_summary,dict_feed)
                train_writer.add_summary(train_summary,step)

                val_iter = 0
                val_loss = 0
                val_acc = 0
                valid_data.new_epoch()
                while val_iter<valid_data.num_batches:
                    val_iter = val_iter + 1
                    valid_batch = getValBatch()
                    valid_batch_data, valid_batch_len = processBatch(valid_batch[0])
                    valid_batch_label = valid_batch[1]
                    valid_dict = {}
                    valid_dict[input_data] = valid_batch_data
                    valid_dict[labels] = valid_batch_label
                    valid_dict[batchLength] = valid_batch_len

                    val_loss = val_loss + sess.run(loss, valid_dict)
                    val_acc = val_acc + sess.run(accuracy, valid_dict)

                print("val_loss:{}".format(val_loss/valid_data.num_batches))
                print("val_acc:{}".format(val_acc/valid_data.num_batches))

                valid_summary =  tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy", simple_value=val_acc/valid_data.num_batches), 
                    tf.Summary.Value(tag="Loss", simple_value=val_loss/valid_data.num_batches), 
                    ])
                valid_writer.add_summary(valid_summary, step)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import math

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logs_path = 'log_softmax_relu_dropout/' # logging path
batch_size = 100 # batch size while performing traing 
learning_rate = 0.003 # Learning rate 
training_epochs = 10 # traing epoch 
display_epoch = 1

# dataPath = "temp/"

# if not os.path.exists(dataPath):
#     os.makedirs(dataPath)
# load 
# train_set_x      = train_set.drop(columns=['label'])
# train_set_x_orig = np.array(train_set_x)
# train_set_y_orig = np.array(train_set['label'][:])
# test_set_x_orig = np.array(test_set)
# X_train_orig, Y_train_orig, X_test_orig = load_dataset()

# mnist = input_data.read_data_sets(dataPath, one_hot=True) # MNIST dataset to be downloaded 
X = tf.placeholder(tf.float32, [None, 784], name='InputData') # mnist data image of shape 28*28=784
XX = tf.reshape(X, [-1, 784]) # reshape input
Y_ = tf.placeholder(tf.float32, [None, 10], name='LabelData') # 0-9 digits recognition => 10 classes

lr = tf.placeholder(tf.float32) # Learning rate 
pkeep = tf.placeholder(tf.float32) # dropout probablity 

L = 200 # number of neurons in layer 1
M = 100 # number of neurons in layer 2
N = 60 # number of neurons in layer 3
O = 30 # number of neurons in layer 4

W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1)) # Initialize random weights for the hidden layer 1 
B1 = tf.Variable(tf.ones([L])) # Bias vector for layer 1

W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1)) # Initialize random weights for the hidden layer 2 
B2 = tf.Variable(tf.ones([M])) # Bias vector for layer 2

W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1)) # Initialize random weights for the hidden layer 3 
B3 = tf.Variable(tf.ones([N])) # Bias vector for layer 3

W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1)) # Initialize random weights for the hidden layer 4
B4 = tf.Variable(tf.ones([O])) # Bias vector for layer 4

W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1)) # Initialize random weights for the hidden layer 5 
B5 = tf.Variable(tf.ones([10])) # Bias vector for layer 5

XX = tf.reshape(X, [-1, 28*28])
pkeep = tf.placeholder(tf.float32)

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1) # Output from layer 1
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2) # Output from layer 2
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3) # Output from layer 3
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4) # Output from layer 4
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5 # computing the logits
Y = tf.nn.softmax(Ylogits) # output from layer 5

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_) # final outcome using softmax cross entropy 
cost_op = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimization op (backprop)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

# Initialize the variables (i.e. assign their default value)
init_op = tf.global_variables_initializer()

# Construct model and encapsulating all ops into scopes, making Tensorboard's Graph visualization more convenient
# Create a summary to monitor cost tensor
tf.summary.scalar("cost", cost_op)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    trainXYOrig = train.values
    m = trainXYOrig.shape[0]
    trainXOrig = trainXYOrig[:, 1:].reshape(-1, 784)  # m * height * width * 1
    trainX = trainXOrig / 255                               # 0-1
    
    trainYOrig = trainXYOrig[:, 0]
    trainY = np.eye(10)[trainYOrig, :]                      # one-hot
    
    testXOrig = test.values.reshape(-1, 784)
    testX = testXOrig / 255
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        # batch_count = int(m/batch_size)
        for i in range(m):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000 
            e = i % batch_size
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-e/decay_speed)
            _, summary = sess.run([train_op, summary_op], {X: trainX, Y_: trainY, pkeep: 0.75, lr: learning_rate})
            writer.add_summary(summary, epoch * m + i)
        print("Epoch: ", epoch)
           
    # print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 0.75}))
    print("done")


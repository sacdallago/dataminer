
# coding: utf-8

# In[ ]:

'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''


# In[ ]:

import tensorflow as tf
from sklearn import cross_validation
from sklearn import metrics as mt
from sklearn import utils as ut
import gc
import csv
import numpy as np
import os.path as path
from os import listdir
from PIL import Image


# In[ ]:

def dataSplit(array, size):
    split = [element['id'] for element in ut.shuffle(array, n_samples=size, random_state=37)]
    return [element for element in array if element['id'] in split], [element for element in array if element['id'] not in split]

def splitPositiveNegative(array, positiveClass):
    return [element for element in array if positiveClass in element['labels_raw']], [element for element in array if positiveClass not in element['labels_raw']]

def proportionalDataSplit(positive, negative, positiveSize, size):
    positiveSize = np.floor(positiveSize*size).astype(int)
    negativeSize = (size-positiveSize).astype(int)
    
    positiveSplit = [element['id'] for element in ut.shuffle(positive, n_samples=positiveSize, random_state=37)]
    negativeSplit = [element['id'] for element in ut.shuffle(negative, n_samples=negativeSize, random_state=37)]
    
    return [element for element in positive if element['id'] in positiveSplit], [element for element in positive if element['id'] not in positiveSplit], [element for element in negative if element['id'] in negativeSplit], [element for element in negative if element['id'] not in negativeSplit]
                     
def dataAndLabels(array, negative = None):
    if negative is None:
        return [np.array(Image.open(path.join(imagesDir, element['id']))) for element in array], [element['labels'] for element in array]
    elif type(negative) is type([]):
        return [np.array(Image.open(path.join(imagesDir, element['id']))) for element in array] + [np.array(Image.open(path.join(imagesDir, element['id']))) for element in negative], [[0,1]]*len(array) + [[1,0]]*len(negative)
    else:
        return


# In[ ]:

with open('./photo_to_levels.csv') as f:
    food_to_label = []
    for row in csv.DictReader(f, skipinitialspace=True):
        element = {}
        for k, v in row.items():
            if k == "id":
                element['id'] = str(v) + ".jpg"
            elif k == "labels":
                labels_raw = np.array(str(v).split(' '))
                labels = [0] * 9
                labels_int = []
                try:
                    for lb in labels_raw:
                        labels[int(str(lb))] = 1
                        labels_int.append(int(lb))
                except ValueError:
                    print "Failure with value", lb, "labels lenght", len(labels_raw), "content:", v
                element['labels'] = labels
                element['labels_raw'] = labels_int
            else :
                print "No idea what you just passed!"
        
        if len(element['labels_raw']) is not 0:
            food_to_label.append(element)
        else:
            print "Picture", element['id'], "has no labels and is being ignored!"

if len(set([element['id'] for element in food_to_label])) != len(food_to_label):
    print('something\'s wrong!')


# In[ ]:

proportions = []
for lb in range(9):
    l = len([element for element in food_to_label if lb in element['labels_raw']])/float(len(food_to_label))
    print "Label", lb, "is present at", int(l*100), "% with respect to all other labels"
    proportions.append(l)


# In[ ]:

# Data dir
imagesDir = './data/SampleFoodClassifier_Norm_100'

# Filter out images which might not be present in the folder but are present in the csv file
files = [f for f in listdir(imagesDir) if path.isfile(path.join(imagesDir, f))]
food_to_label = [element for element in food_to_label if element['id'] in files]

del files[:]
gc.collect()

print "The new length of the data is", len(food_to_label)

# Parameters
test_size = 500
learning_rate_start= .001
training_size = 100
training_iters = 100
dropout = 0.75 # Dropout, probability to keep units

# Network Parameters
# !! Images: 100x100 RGB = 100, 100, 3
w, h, channels = [100, 100, 3]
n_classes = 2
print "Width, Height and channels:", w, h, channels, ". Number of classes:", n_classes

# tf Graph input
x = tf.placeholder(tf.float32, [None, w, h, channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# In[ ]:

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print "PLEASE MODIFY WD1 TO", conv2.get_shape().as_list()[1], "*",conv2.get_shape().as_list()[2], "*64"

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# In[ ]:

# Store layers weight & bias
sdev= 0.01
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=sdev)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=sdev)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([25*25*64, 3000], stddev=sdev)),
    'wd2': tf.Variable(tf.truncated_normal([3000, 1024], stddev=sdev)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=sdev))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=sdev)),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=sdev)),
    'bd1': tf.Variable(tf.truncated_normal([3000], stddev=sdev)),
    'bd2': tf.Variable(tf.truncated_normal([1024], stddev=sdev)),
    'out': tf.Variable(tf.truncated_normal([n_classes], stddev=sdev))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))

# optimizer without adapted learning_rate
#optimizer = tf.train.AdamOptimizerOptimizer(learning_rate=learning_rate).minimize(cost)

#optimizer with adapted learning_rate
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(learning_rate_start, step, 1, 0.9999)

optimizer = tf.train.AdamOptimizer(rate).minimize(cost, global_step=step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_p = tf.argmax(pred, 1)


# Gives an array of arrays, where each position represents % of belonging to respective classs. Eg: a[0.34, 0.66] --> class 0 : 34%, class 1: 66%
# classes = tf.nn.softmax(pred)
classes = tf.nn.softmax(pred)


def label_class(x):
    for i in range(0,len(x)):
        print i, ":", x[i]

# Initializing the variables
init = tf.initialize_all_variables()


# In[ ]:

# save the models
saveDir = './tensorflow/one_against_all'
saver = tf.train.Saver()


# In[ ]:

for currentLabel in range(len(proportions)):
    print "Training model for", currentLabel, "which is represented by", proportions[currentLabel]
    positive, negative = splitPositiveNegative(food_to_label, currentLabel)
    
    # Get some test samples
    test1, positive, test0, negative = proportionalDataSplit(positive, negative, proportions[currentLabel], test_size)
    
    positiveSize = np.floor(proportions[currentLabel]*training_size).astype(int)
    negativeSize = (training_size-positiveSize).astype(int)
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Keep training until reach max iterations
        for epoch in range(training_iters):
            if len(positive) < positiveSize or len(negative) < negativeSize:
                del ix[:]
                del iy[:]
                del batch1[:]
                del batch0[:]
                break
            # Fit training using batch data
            print "Loading batch...",
            batch1, positive, batch0, negative = proportionalDataSplit(positive, negative, proportions[currentLabel], training_size)
            ix, iy = dataAndLabels(batch1, batch0)
            print "bactch loaded!"

            print "Running optimizer...",
            sess.run(optimizer, feed_dict={x: ix, y: iy, keep_prob: 1.})
            print "done!"
            # Compute average loss
            loss, acc = sess.run([cost, accuracy], feed_dict={x: ix, y: iy, keep_prob: 1.})
            # Display logs per epoch step
            print "Iter " + str(epoch) + ", Minibatch Loss= " +                       "{:.6f}".format(loss) + ", Training Accuracy= " +                       "{:.5f}".format(acc)
            del ix[:]
            del iy[:]
            del batch1[:]
            del batch0[:]
        print "Optimization Finished!"
        
        save_path = saver.save(sess, "/tmp/label-" + str(currentLabel) + ".ckpt")
        print("Model saved in file: %s" % save_path)

        runs = 0
        acc = 0.
        y_pred = []
        class_pred = []
        test = test1 + test0
        y_test = [[0,1]]*len(test1) + [[1,0]]*len(test0)
        for i in range(0, test_size, 30):
            if i+30 < test_size:
                x_test, _ = dataAndLabels(test[i:i+30])
                val_accuracy, y_pred_i, cls = sess.run([accuracy, y_p, classes], feed_dict={x: x_test, y: y_test[i:i+30], keep_prob: 1.})
            else:
                x_test, _ = dataAndLabels(test[i:])
                val_accuracy, y_pred_i, cls = sess.run([accuracy, y_p, classes], feed_dict={x: x_test, y: y_test[i:], keep_prob: 1.})
            acc += val_accuracy
            y_pred.extend(y_pred_i)
            class_pred.extend(cls)
            runs += 1
            print "Partial testing accuracy:", acc/runs

        #metrics
        print "Validation accuracy:", acc/runs
        y_true = np.argmax(y_test,1)
        print "Precision for each class:"
        label_class(mt.precision_score(y_true, y_pred, average=None))
        print "Recall for each class:"
        label_class(mt.recall_score(y_true, y_pred, average=None))
        print "F1_score for each class:"
        label_class(mt.f1_score(y_true, y_pred, average=None))
        print "confusion_matrix"
        print mt.confusion_matrix(y_true, y_pred)
        fpr, tpr, tresholds = mt.roc_curve(y_true, y_pred)


# In[ ]:

for i in range(len(y_test)):
    print "For", i, "as", y_test[i]
    for j in range(2):
        print "\t", j, "@", class_pred[i][j]*100
    print "\n"


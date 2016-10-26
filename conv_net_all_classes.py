
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
                    i=0 #dummy thing to prevent the print spam below
                    #print "Failure with value", lb, "labels lenght", len(labels_raw), "content:", v
                element['labels'] = labels
                element['labels_raw'] = labels_int
            else :
                print "No idea what you just passed!"
        
        if len(element['labels_raw']) is not 0:
            food_to_label.append(element)
        #else:
        #    print "Picture", element['id'], "has no labels and is being ignored!"

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
training_size = 30
training_iters = 20
threshold= 0.1 #threshold for accepting a label (in terms of probability; note: all probabilities sum up to 1)
dropout = 0.75 # Dropout, probability to keep units
sdev= 0.01 #Stddev of the initiliazed variables. Too small causes stationarity of the modell (NOT <0.01!), too large gives large variety

seed=ut.resample(np.linspace(1,1000,1000,dtype=int), n_samples=1)[0]
print "Seed used for data split:", seed

# Network Parameters
# !! Images: 100x100 RGB = 100, 100, 3
w, h, channels = [100, 100, 3]
n_classes = 9
print "Width, Height and channels:", w, h, channels, ". Number of classes:", n_classes

# tf Graph input
x = tf.placeholder(tf.float32, [None, w, h, channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# In[5]:

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
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# In[6]:

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=sdev, seed=seed)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=sdev, seed=seed)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([25*25*64, 1024], stddev=sdev, seed=seed)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=sdev, seed=seed))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=sdev, seed=seed)),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=sdev, seed=seed)),
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=sdev, seed=seed)),
    'out': tf.Variable(tf.truncated_normal([n_classes], stddev=sdev, seed=seed))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits (pred, y))

#optimizer with adapted learning_rate
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(learning_rate_start, step, 1, 0.9999)

optimizer = tf.train.AdamOptimizer(rate).minimize(cost, global_step=step)

# Gives an array of arrays, where each position represents % of belonging to respective classs. Eg: a[0.34, 0.66] --> class 0 : 34%, class 1: 66%
classes = tf.nn.softmax(pred)
y_p = tf.cast(tf.greater(classes,threshold),dtype="float")

#Evaluate model
correct_pred = tf.cast(tf.equal(y_p,y),dtype="float")
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def label_class(x):
    for i in range(0,len(x)):
        print i, ":", x[i]

# Initializing the variables
init = tf.initialize_all_variables()


# In[7]:

def dataSplit(array, size,seed=seed):
    split = [element['id'] for element in ut.shuffle(array, n_samples=size, random_state=seed)]
    return [element for element in array if element['id'] in split], [element for element in array if element['id'] not in split]

def dataAndLabels(array):
    return [np.array(Image.open(path.join(imagesDir, element['id']))) for element in array], [element['labels'] for element in array]


# In[8]:

# Get out a sample of data for testing
## IMPORTANT: RUN ONLY ONCE!
test, food_to_label = dataSplit(food_to_label, test_size)


# In[28]:

def evaluation(y_pred,y_test, n_classes):
    
    precision0 = []
    precision1 = []
    recall0 = []
    recall1 = []
    F10 = []
    F11 = []
    tn = []
    fn = []
    tp = []
    fp = []
    
    for i in range(n_classes):
        y_pred_i = [float(el[i]) for el in y_pred]
        y_test_i = [float(el[i]) for el in y_test]
        tn.append(np.sum(np.logical_and(np.equal(y_pred_i,0),np.equal(y_test_i,0))))
        fn.append(np.sum(np.logical_and(np.equal(y_pred_i,0),np.equal(y_test_i,1))))
        tp.append(np.sum(np.logical_and(np.equal(y_pred_i,1),np.equal(y_test_i,1))))
        fp.append(np.sum(np.logical_and(np.equal(y_pred_i,1),np.equal(y_test_i,0))))
        try:
            precision0.append(float(tn[i])/(fn[i]+tn[i]))
        except ZeroDivisionError:
            precision0.append(1)
        try: 
            precision1.append(float(tp[i])/(fp[i]+tp[i]))
        except ZeroDivisionError:
            precision0.append(1)
        try:
            recall0.append(float(tn[i])/(fp[i]+tn[i]))
        except ZeroDivisionError:
            recall0.append(1)
        try:
            recall1.append(float(tp[i])/(tp[i]+fn[i]))
        except ZeroDivisionError:
            recall1.append(1)
        try:
            F10.append(2*precision0[i]*recall0[i]/(precision0[i]+recall0[i]))
        except ZeroDivisionError:
            F10.append(1)
        try:
            F11.append(2*precision1[i]*recall1[i]/(precision1[i]+recall1[i]))
        except:
            F11.append(1)
    
    print
        
    print 'Precision:'
    print "\t" + "0" + "\t" + "\t" + "1"
    print "---------------------------------------"
    for i in range(n_classes):
        print str(i) + ":"+ "\t" + "{:.2f}".format(precision0[i]*100) + "\t" + "%\t" + "{:.2f}".format(precision1[i]*100) + "\t"+ "%"
    print "---------------------------------------"
    print "avg:" + "\t" + "{:.2f}".format(100*sum(precision0)/float(len(precision0))) + "\t"+ "%\t" + "{:.2f}".format(100*sum(precision1)/float(len(precision1))) + "\t" + "%"
    
    print
    print 'Recall:'
    print "\t" + "0" + "\t" + "\t" + "1" 
    print "---------------------------------------"
    for i in range(n_classes):
        print str(i) + ":"+ "\t" + "{:.2f}".format(100*recall0[i]) + "\t" + "%\t" + "{:.2f}".format(100*recall1[i]) + "\t" + "%"
    print "---------------------------------------"
    print "avg:" + "\t" + "{:.2f}".format(100*sum(recall0)/float(len(recall0))) + "\t" + "%\t" + "{:.2f}".format(100*sum(recall1)/float(len(recall1))) + "\t" + "%"
    
    print
    print 'F1-Score:'
    print "\t" + "0" + "\t" + "\t" + "1" 
    print "---------------------------------------"
    for i in range(n_classes):
        print str(i) + ":"+ "\t" + "{:.2f}".format(100*F10[i]) + "\t" + "%\t" + "{:.2f}".format(100*F11[i]) + "\t" + "%"
    print "---------------------------------------"
    print "avg:" + "\t" + "{:.2f}".format(100*sum(F10)/float(len(F10))) + "\t" + "%\t" + "{:.2f}".format(100*sum(F11)/float(len(F11))) + "\t" + "%"


# In[29]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Keep training until reach max iterations
    for epoch in range(training_iters):
        if len(food_to_label) < training_size:
            del ix[:]
            del iy[:]
            del batch[:]
            break
        # Fit training using batch data
        print "Loading batch...",
        batch, food_to_label = dataSplit(food_to_label, training_size)
        ix, iy = dataAndLabels(batch)
        print "batch loaded!"
        
        print "Running optimizer...",
        sess.run(optimizer, feed_dict={x: ix, y: iy, keep_prob: 1.})
        print "done!"
        # Compute average loss
        loss, acc = sess.run([cost, accuracy], feed_dict={x: ix, y: iy, keep_prob: 1.})
        # Display logs per epoch step
        print "Iter " + str(epoch) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc)
        del ix[:]
        del iy[:]
        del batch[:]
    print "Optimization Finished!"

    runs = 0
    acc = 0.
    y_pred = []
    y_test = []
    class_pred = []
    for i in range(0, test_size, 30): #test batch size = 30
        if i+30 < test_size:
            x_test, y_test_partial = dataAndLabels(test[i:i+30])
        else:
            x_test, y_test_partial = dataAndLabels(test[i:])
        val_accuracy, y_pred_i, cls = sess.run([accuracy, y_p, classes], feed_dict={x: x_test, y: y_test_partial, keep_prob: 1.})
        acc += val_accuracy
        y_test.extend(y_test_partial)
        y_pred.extend(y_pred_i)
        class_pred.extend(cls)
        runs += 1
        print "Partial testing accuracy:", acc/runs
    
    #metrics
    print "Validation accuracy:", acc/runs
    evaluation(y_test, y_pred, n_classes)
    #Down here things go nuts, VALIDATION was reviewwed to be reviewed!
    #print "Precision for each class:"
    #print "Recall for each class:"
    #label_class(mt.recall_score(y_test, y_pred, average=None))
    #print "F1_score for each class:"
    #label_class(mt.f1_score(y_test, y_pred, average=None))
    #print "confusion_matrix"
    #print mt.confusion_matrix(y_test, y_pred)
    #fpr, tpr, tresholds = mt.roc_curve(y_true, y_pred)


# In[23]:

for i in range(len(y_test)):
    print "For", i, "as", y_test[i]
    for j in range(9):
        print "\t", j, "@", class_pred[i][j]*100
    print "\n"


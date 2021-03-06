import tensorflow as tf
from sklearn import cross_validation
from sklearn import metrics as mt
from sklearn import utils as ut
import csv
import numpy as np
import os.path as path
from PIL import Image
from skimage.color import rgb2gray

# Load data
featuresDir = './data/SampleFoodClassifier_Norm_300'

with open('./sample_food_no_food.csv') as f:
    food_no_food = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    

data_ids = [element['id'] for element in food_no_food]

labels = [int(element['is_food']) for element in food_no_food]
#data = [rgb2gray(np.array(Image.open(path.join(featuresDir, element)))) for element in data_ids]
data = [np.array(Image.open(path.join(featuresDir, element))) for element in data_ids]
seed=ut.resample(np.linspace(1,1000,1000,dtype=int), n_samples=1)[0]

# Split training data in a train set and a test set. The test set will containt 20% of the total
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=.25,random_state=seed)

# Parameters
#learning_rate = 0.001
learning_rate_start= 0.0001
training_size = 20
training_split= 0.8 #split into training_split 1 or 0 and (1-training_split) 0 
training_iters_max = 100
training_freq= 2       #how often is each picture trained (on average!)
sdev= 0.01 #for variable initialization the higher the stddev, the more iterations are needed
#stddev too small leads to stationary variables (don't take it smaller than 0.01!)!

training_size_1=np.floor(training_split*training_size).astype(int)
training_size_0=(training_size-training_size_1).astype(int)

# Network Parameters
w, h, channels = data[0].shape
print "Width, Height and channels:", w, h, channels
n_classes = len(set(y_train))
dropout = 0.75 # Dropout, probability to keep units

# print 'Input vector size', n_input, 'train shape', np.array(x_train).shape , 'number of classes', n_classes

# tf Graph input
x = tf.placeholder(tf.float32, [None, w, h, channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

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


# Store layers weight & bias

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=sdev, seed=1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=sdev, seed=2)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([25*25*64, 1024], stddev=sdev, seed=3)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=sdev, seed=4))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=sdev, seed=5)),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=sdev, seed=6)),
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=sdev, seed=7)),
    'out': tf.Variable(tf.truncated_normal([n_classes], stddev=sdev, seed=8))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

#optimizer without adapted learning_rate
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#optimizer with adapted learning_rate
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(learning_rate_start, step, 1, 0.9999)

optimizer = tf.train.AdamOptimizer(rate).minimize(cost, global_step=step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_p = tf.argmax(pred, 1)

def label_class(x):
    for i in range(0,len(x)):
        print i, ":", x[i]

# Initializing the variables
init = tf.initialize_all_variables()


y_train_temp = []

for element in y_train:
    temp = [0]*len(set(y_train))
    temp[element] = 1
    y_train_temp.append(temp)
    
y_train = np.reshape(y_train_temp,(len(y_train_temp), -1))

y_test_temp = []

for element in y_test:
    temp = [0]*len(set(y_test))
    temp[element] = 1
    y_test_temp.append(temp)
    
y_test = np.reshape(y_test_temp,(len(y_test_temp), -1))

# Proportional sampling from both classes, get features for 0 and 1 each
y_help=np.array([el[1] for el in y_train])
y_index_0 = np.where(y_help==0)[0]
y_index_1 = np.where(y_help==1)[0]

x_0 = [x_train[index] for index in y_index_0]
x_1 = [x_train[index] for index in y_index_1]

#y batch looks always the same for if using proportional sampling
iy=np.vstack(([[1,0]]*training_size_0,[[0,1]]*training_size_1))


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    dummy_0 = len(x_0)
    dummy_1 = len(x_1)
    
    # Keep training until reach max iterations
    for epoch in range(training_iters_max):
        
        if (dummy_0 < training_size_0 or dummy_1 < training_size_1):
            break
            
        # Fit training using batch data first training_size_0 data are 0 rest random
        x_batch_0 = ut.resample(x_0, n_samples=training_size_0,replace=False, random_state=seed+epoch)
        x_batch_1 = ut.resample(x_1, n_samples=training_size_1,replace=False, random_state=seed+epoch)
        ix = x_batch_0+x_batch_1 #is for concatenating the lists, no addition done here
        
        dummy_0 = dummy_0 - training_size_0/training_freq
        dummy_1 = dummy_1 - training_size_1/training_freq
        
        #ix = ut.shuffle(x_train, n_samples=training_size, random_state=epoch)
        #iy = ut.shuffle(y_train, n_samples=training_size, random_state=epoch)

        sess.run(optimizer, feed_dict={x: ix, y: iy, keep_prob: 1.})
        
        # Compute average loss
        loss, acc = sess.run([cost, accuracy], feed_dict={x: ix, y: iy, keep_prob: 1.})
        # Display logs per epoch step
        print "Iter " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
    print "Optimization Finished!"
        
    runs = 0
    acc = 0.
    y_pred = []
    for i in range(0, len(y_test), 10):
        if i+10 < len(y_test):
            val_accuracy, y_pred_i = sess.run([accuracy, y_p], feed_dict={x: x_test[i:i+10], y: y_test[i:i+10], keep_prob: 1.})
            acc += val_accuracy
            y_pred.extend(y_pred_i)
        else:
            val_accuracy, y_pred_i = sess.run([accuracy, y_p], feed_dict={x: x_test[i:], y: y_test[i:], keep_prob: 1.})
            acc += val_accuracy
            y_pred.extend(y_pred_i)
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
    print "confusion_matrix:"
    print mt.confusion_matrix(y_true, y_pred)
    fpr, tpr, tresholds = mt.roc_curve(y_true, y_pred)
    print "seed used for splitting:", seed



    

# imports
import numpy as np
import cv2
import tensorflow as tf
#import matplotlib.pyplot as plt
import os

from cv2 import saliency_MotionSaliency
from sklearn.model_selection import StratifiedShuffleSplit


# image import directories
good_nail_dir = ".../nailgun/good/"
bad_nail_dir = ".../nailgun/bad/"

# model and checkpoint paths
model_dir = ".../model/"
checkpoint_path = model_dir + "intermediate_checkpoint.ckpt"
# intermediate checkpoint epoch path
checkpoint_epoch_path = checkpoint_path+".epoch"
# final model path
final_model_path = model_dir + "my_model"

# list to load the images
good_nails = []
bad_nails = []


# return random instances of the data and labels provided of size "batch_size"
def fetch_batch(feature_set, labels, batch_size, epoch):
    np.random.seed(epoch)
    split = StratifiedShuffleSplit(n_splits=1, test_size=batch_size / len(feature_set))
    for train_index, test_index in split.split(feature_set, labels):
        train_data, test_data = feature_set[train_index], feature_set[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
    return (test_data, test_labels)


# process and linearize the images
def data_preprocess(good_nails, bad_nails):
    good_nails_resize = []  # list to store the resized good images
    bad_nails_resize = []  # list to store the resized bad images
    for i in range(len(good_nails)):
        temp = good_nails[i,235:985,600:1350]  # crop the unnecessary boundry
        ret, thresh = cv2.threshold(temp, 127, 255, cv2.THRESH_TRUNC)  # threshold image generation
        thresh = (thresh - np.mean(thresh)) / np.std(thresh)
        #temp_a = cv2.resize(cv2.GaussianBlur(thresh, (5, 5), 0.1), (500, 500))
        #temp_b = cv2.resize(cv2.GaussianBlur(temp_a, (5, 5), 0.1), (128, 128))
        temp_a = cv2.resize(thresh, (500, 500))
        temp_b = cv2.resize(temp_a, (250, 250))
        good_nails_resize.append(temp_b)
        temp = bad_nails[i,235:985,600:1350]
        ret, thresh = cv2.threshold(temp, 127, 255, cv2.THRESH_TRUNC)
        thresh = (thresh - np.mean(thresh)) / np.std(thresh)
        #temp_a = cv2.resize(cv2.GaussianBlur(thresh, (5, 5), 0.1), (500, 500))
        #temp_b = cv2.resize(cv2.GaussianBlur(temp_a, (5, 5), 0.1), (128, 128))
        temp_a = cv2.resize(thresh, (500, 500))
        temp_b = cv2.resize(temp_a, (250, 250))
        bad_nails_resize.append(temp_b)
    good_nails_resize = np.array(good_nails_resize)
    bad_nails_resize = np.array(bad_nails_resize)
    return good_nails_resize.reshape(-1,250*250), bad_nails_resize.reshape(-1,250*250)

# return train data, test data, train labels and test labels
def train_test_split(good_nails, bad_nails,test_size):
    data = np.r_[good_nails, bad_nails]
    label = np.r_[np.ones(len(good_nails)), np.zeros(len(bad_nails))]
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = label[train_index], label[test_index]
    return train_data, train_labels, test_data, test_labels



print("Data set is loding.....")
for nail in os.listdir(good_nail_dir):
    good_nails.append(cv2.imread(good_nail_dir + nail, cv2.IMREAD_GRAYSCALE))
for nail in os.listdir(bad_nail_dir):
    bad_nails.append(cv2.imread(bad_nail_dir + nail, cv2.IMREAD_GRAYSCALE))

good_nails = np.array(good_nails)
bad_nails = np.array(bad_nails)
good_nails_resize, bad_nails_resize = data_preprocess(good_nails, bad_nails)
train_data, train_labels, test_data, test_labels = train_test_split(good_nails_resize, bad_nails_resize, 0.05)
#plt.imshow(test_data[0].reshape(128,128))
#plt.show()
print()
print("Augmenting the data set....")
# build graph for data augmentation
height = 250
width = 250
channel = 1
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None,height*width*channel], name="X")
y = tf.placeholder(tf.int64, [None], name="y")

reshape = tf.reshape(X,[-1,height,width,channel])
v_flip = tf.concat([tf.image.flip_up_down(reshape),reshape],axis=0)
v_flip_label = tf.concat([y,y],axis=0)
h_flip = tf.concat([tf.image.flip_left_right(v_flip),v_flip],axis=0)
h_flip_label = tf.concat([v_flip_label,v_flip_label],axis=0)
rotated_20 = tf.contrib.image.rotate(h_flip,20,"NEAREST")
rotated_40 = tf.contrib.image.rotate(rotated_20,20,"NEAREST")
rotated_60 = tf.contrib.image.rotate(rotated_40,20,"NEAREST")
rotated_80 = tf.contrib.image.rotate(rotated_60,20,"NEAREST")
rotated_100 = tf.contrib.image.rotate(rotated_80,20,"NEAREST")
rotated_120 = tf.contrib.image.rotate(rotated_100,20,"NEAREST")
rotated_140 = tf.contrib.image.rotate(rotated_120,20,"NEAREST")
rotated_160 = tf.contrib.image.rotate(rotated_140,20,"NEAREST")
rotated_180 = tf.contrib.image.rotate(rotated_160,20,"NEAREST")
all_combine = tf.concat([h_flip,rotated_20,rotated_40,rotated_60,
                        rotated_80,rotated_100,rotated_120,
                        rotated_140,rotated_160,rotated_180],axis=0)
all_combine_reshaped = tf.reshape(all_combine,[-1,250*250])
all_labels = tf.concat([h_flip_label,h_flip_label,h_flip_label,
                       h_flip_label,h_flip_label,h_flip_label,
                       h_flip_label,h_flip_label,h_flip_label,h_flip_label],axis=0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    aug_train_data,aug_train_labels = sess.run([all_combine_reshaped,
                                                all_labels],
                                                feed_dict={X: train_data,
                                                            y: train_labels})

# Graph definition
height = 250
width = 250
channel = 1
learning_rate = 0.001
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, height * width * channel], name="X")
y = tf.placeholder(tf.int64, [None], name="y")
training = tf.placeholder_with_default(False, shape=(), name="training_variable")
# placeholder for learning rate scheduling
# learning_rate = tf.placeholder_with_default(0.01,shape=(), name= "learning_rate")
global_step_tensor = tf.Variable(0, trainable=False, name="global_step")

input_layer = tf.reshape(X, [-1, height, width, channel],name="input")
# layer_1: Convolution
conv1 = tf.layers.conv2d(inputs=input_layer,
                         filters=96,
                         strides=4,
                         kernel_size=[9, 9],
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv_1")

# layer_2: Maxpool
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2],
                                strides=2,
                                name="pool_1")

# layer_3: Convolution
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=128,
                         kernel_size=[5, 5],
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv_2")
                         #trainable=False)

# layer_4: Maxpool
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2],
                                strides=2,
                                name="pool_2")
# layer_5: Convolution
conv3 = tf.layers.conv2d(inputs=pool2,
                         filters=256,
                         kernel_size=[3, 3],
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv_3")

# layer_6: Maxpool
pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                pool_size=[2, 2],
                                strides=2,
                                name="pool_3")

# layer_7: Convolution
conv4 = tf.layers.conv2d(inputs=pool3,
                         filters=400,
                         kernel_size=[3, 3],
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv_4")

# layer_8: Maxpool
pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                pool_size=[2, 2],
                                strides=2,
                                name="pool_4")

# flatten the layer_8 to apply to ANN
poolFlat = tf.reshape(pool4, [-1, 3 * 3 * 400])
#
## layer_9: Fully connected
dense1 = tf.layers.dense(inputs=poolFlat,
                         units=4000,
                         activation=tf.nn.relu,
                         name="dense_1")

## layer_10: Dropout
dropout1 = tf.layers.dropout(dense1,
                             rate=0.3,
                             training=training,
                             name="dropout_1")

# layer_11: Fully connected
dense2 = tf.layers.dense(inputs=dropout1,
                         units=500,
                         activation=tf.nn.relu,
                         name="Dense_2")

# layer_12: Dropout
dropout2 = tf.layers.dropout(dense2,
                             rate=0.3,
                             training=training,
                             name="Dropout_2")

# layer_13: Dense
dense3 = tf.layers.dense(inputs=dropout2,
                         units=100,
                         activation=tf.nn.relu,
                         name="Dense_3")

# layer_14: Dropout
dropout3 = tf.layers.dropout(dense3,
                             rate=0.3,
                             training=training,
                             name="Dropout_3")

# output layer
logits = tf.layers.dense(inputs=dropout3, units=2,name="logits")
prediction = tf.argmax(tf.nn.softmax(logits),1,name="predict")
#classes = tf.argmax(logits,axis=1)
#
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=y,
                                                          name="xentropy")
loss = tf.reduce_mean(xentropy,name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, global_step=global_step_tensor)

correct = tf.nn.in_top_k(logits, y, 1,name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32),name="accuracy")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

print("Training started....")
n_epoch = 500
batch_size = 100
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path) as f:
            start_epoch = int(f.read())
        print("Training was interupted resuming from epoch ", start_epoch)
        saver.restore(sess,checkpoint_path)
    else:
        start_epoch = 0
        init.run()
    for epoch in range(start_epoch,n_epoch):
        X_batch, y_batch = fetch_batch(aug_train_data,aug_train_labels, batch_size,epoch)
        sess.run(training_op, feed_dict={X:X_batch,
                                             y:y_batch,
                                             training:True})
        # Early stopping implementation
        #X_batch_val, y_batch_val = fetch_batch(cross_v, cross_v_labels, batch_size,epoch)
        #new_loss = sess.run(loss,feed_dict={X: X_batch_val,y: y_batch_val})
        #if new_loss<old_loss:
        #    saver.save(sess,final_model_path)
        #    old_loss = new_loss
        #    best_epoch = epoch
        #    early_stop_count=0
        #else:
        #    early_stop_count+=1
        #    if early_stop_count>100:
        #        print("Early stopping satisfied with best epoch ",best_epoch)
        #        current_train_loss, current_train_accu = sess.run([loss,accuracy],
        #                                                              feed_dict={X:X_batch,
        #                                                                        y:y_batch})
        #        current_val_loss, current_val_accu = sess.run([loss,accuracy],
        #                                                        feed_dict={X:X_batch_val,
        #                                                                    y:y_batch_val})
        #        print("Current train loss: ",current_train_loss,
        #                  " Current train accuracy: ", current_train_accu)
        #        print()
        #        print("Current val loss: ",current_val_loss,
        #                  " Current val accuracy: ", current_val_accu)
        #        print()
        #        print("Best model loss: ", old_loss)
        #        break
        if epoch%10==0:
            if epoch%30==0:
                print("Saving checkpoint for epoch ",epoch)
                saver.save(sess,checkpoint_path)
                with open(checkpoint_epoch_path,"wb") as f:
                    f.write(b'%d' % (epoch+1))
            train_loss, train_accuracy = sess.run([loss, accuracy],
                                                    feed_dict={X:X_batch, y:y_batch})
            print("Epoch: ",epoch," Training loss: ",train_loss," Training accuracy: ",train_accuracy)
            print()
    saver.save(sess,final_model_path)
    test_accu = sess.run(accuracy,feed_dict={X:test_data,y:test_labels})
    print("Test accuracy: ",test_accu)

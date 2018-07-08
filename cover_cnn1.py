# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import cv2
from PIL import Image
from pathlib import Path

# Common imports
import numpy as np
import os
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
    
    
def loadImgData(datadir):
    p = Path(datadir)
    files = list(p.glob('*.png'))
    
    data = []
    label = []
    fnames = []
    for f in files:
        fname = f.name
        label.append(1 if fname.startswith('p') else 0)
        data.append(np.array(Image.open(os.path.join(datadir, fname))))
        fnames.append(fname)
        
    dataset = np.expand_dims(np.stack(data), axis=3).astype(float)
    dataset = np.asarray(dataset, dtype=np.float32)
    label = np.asarray(label, dtype=np.int32)
    print("load data from {}, total:{}".format(datadir, str(len(label))))
    return  dataset, label, fnames
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sklearn.metrics as sk_me

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.convert_to_tensor(features['images'], dtype=tf.float32) #would be [-1, 125, 200, 1]
    
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1] 
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(tensor=pool2, shape=[-1, 99200])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "precision":tf.metrics.precision(
          labels=labels, predictions=predictions["classes"]),
      "recall":tf.metrics.recall(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def myeval():
    # Load training and eval data
#    train_data, train_labels, fnames_train = loadImgData('mydata/img_train')
    eval_data, eval_labels, fnames_eval = loadImgData('mydata/img_valid_s')

    # Create the Estimator
    cover_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="tf_modal/cover1_cnn")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"images": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = cover_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
def mytest(folder):
    # Load training and eval data
#    train_data, train_labels, fnames_train = loadImgData('mydata/img_train')
    dataset, labels, fnames_eval = loadImgData(folder)

    # Create the Estimator
    cover_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="tf_modal/cover1_cnn")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"images": dataset},
      y=labels,
      num_epochs=1,
      shuffle=False)
    predictions = cover_classifier.predict(input_fn=eval_input_fn)
    
    print('gen_resultset...')
    return gen_resultset(zip(fnames_eval, predictions), folder)
    
import math
def showresult(results, onlyerror = False):
    
    if onlyerror:
        results = [r for r in results if (r['FP'] == True or r['FN'] == True)]
        
    res_len = len(results)
    columns = 5
    rows = int(math.ceil(res_len/columns))
    
    fig=plt.figure(figsize=(16, rows*3), dpi= 160)
    fig.subplots_adjust(hspace=0.8, wspace=0.4)

    i = 1
    for r in results:
        ax = fig.add_subplot(rows, columns, i)
        i = i + 1
        ax.set_title(r['title'])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        color = 'red' if r['FP'] or r['FN'] else 'green'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        plt.xlabel(r['sub_title'])
        plt.imshow(Image.open(r['fname']))

    
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()
    
def gen_resultset(prid, imgdir):
    
    print('enter gen_resultset')
    resset = []
    labels = []
    cls = []
    for p in prid:
        fname, pre = p
        labels.append(1 if fname.startswith('p') else 0)
        cls.append(1 if pre['classes'] else 0)
        
        fp = True if pre['classes'] == 1 and fname.startswith('n') else False
        fn =True if pre['classes'] == 0 and fname.startswith('p') else False
        resset.append({'fname':imgdir + '/' + fname,
                       'title': fname,
                       'FP': fp,
                       'FN': fn,
                      'class':pre['classes'],
                      'sub_title':'{0}:{1:.3f}:{2:.3f}'.format(pre['classes'], pre['probabilities'][0], pre['probabilities'][1])})
        
    
    eval1 = {'accuracy':sk_me.accuracy_score(labels, cls),
             'precision':sk_me.precision_score(labels, cls),
           'recall':sk_me.recall_score(labels, cls)}

    return resset, eval1
    
resset, eval1 = mytest('mydata/img_test_s')


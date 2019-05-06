"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
It is very similar to mnist_tutorial_tf.py, which does the same
thing but without a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

from tqdm import tqdm
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

FLAGS = flags.FLAGS
PKLDATA='../df'
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = '../train_dir'
FILENAME = 'mnist.ckpt'
LOAD_MODEL = True


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, train_dir=TRAIN_DIR,
                   filename=FILENAME, load_model=LOAD_MODEL,
                   testing=True, label_smoothing=0.1):
  """
  MNIST CleverHans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param train_dir: Directory storing the saved model
  :param filename: Filename to save model under
  :param load_model: True for load, False for not load
  :param testing: if true, test error is calculated
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """
  tf.keras.backend.set_learning_phase(0)

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  if keras.backend.image_data_format() != 'channels_last':
    raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

  # Create TF session and set as Keras backend session
  sess = tf.Session()
  keras.backend.set_session(sess)

  # Get MNIST test data
  x_train, x_test, x_decode_train, x_decode_test, y_train, y_test = pickle.load(open('../mnist_decoded.pkl'))

  # Get FG data
  adv_train, adv_test, adv_train_decoded, adv_test_decoded = pickle.load(open(PKLDATA+'_decoded.pkl'))

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Define TF model graph
  model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)
  preds = model(x)
  print("Defined TensorFlow model graph.")

  def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
    print('Test accuracy on legitimate examples: %0.4f' % acc)

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'train_dir': train_dir,
      'filename': filename
  }

  rng = np.random.RandomState([2017, 8, 30])
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)

  ckpt = tf.train.get_checkpoint_state(train_dir)
  print(train_dir, ckpt)
  ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap = KerasModelWrapper(model)

  if load_model and ckpt_path:
    saver = tf.train.Saver()
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Model loaded from: {}".format(ckpt_path))
    evaluate()
  else:
    print("Model was not loaded, training from scratch.")
    loss = CrossEntropy(wrap, smoothing=label_smoothing)
    train(sess, loss, x_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng)

  # Calculate training error
  if testing:
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
    report.train_clean_train_clean_eval = acc

  batch = 1000
  x_adv_test = None
  for i in tqdm(range(int(len(x_test) / batch))):
    tmp = sess.run(adv_x, feed_dict={x: x_test[i*batch:(i+1)*batch]})
    if x_adv_test is None:
      x_adv_test = tmp
    else:
      x_adv_test = np.concatenate((x_adv_test, tmp))
  print(x_adv_test[0])


  x_adv_train = None
  for i in tqdm(range(int(len(x_train) / batch))):
    tmp = sess.run(adv_x, feed_dict={x: x_train[i*batch:(i+1)*batch]})
    if x_adv_train is None:
      x_adv_train = tmp
    else:
      x_adv_train = np.concatenate((x_adv_train, tmp))
  def evaluate_adv():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_adv_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
    print('Test accuracy on legitimate examples: %0.4f' % acc)

  evaluate_adv()

  save_list = [x_adv_train, x_adv_test]
  print(x_adv_train.shape)
  print(x_adv_test.shape)
  pickle.dump(save_list, open("./sal.pkl", 'wb'))


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 train_dir=FLAGS.train_dir,
                 filename=FLAGS.filename,
                 load_model=FLAGS.load_model)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('train_dir', TRAIN_DIR,
                      'Directory where to save model.')
  flags.DEFINE_string('filename', FILENAME, 'Checkpoint filename.')
  flags.DEFINE_boolean('load_model', LOAD_MODEL,
                       'Load saved model or train.')
  tf.app.run()

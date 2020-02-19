"""
Single-variable linear regression model. This code is largely taken from:

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

import ml.defaults as defaults
import ml.models.helpers as helpers


def build_train_test_eval(feature, labels, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):
    num_samples = feature.shape[0]

    # Set up the model.
    X = tf.placeholder('float')
    Y = tf.placeholder('float')

    W = tf.Variable(np.random.randn(), name='weight')
    b = tf.Variable(np.random.randn(), name='bias')

    pred = tf.add(tf.mul(X, W), b)

    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * num_samples)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    # Train it.
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            for (x, y) in zip(feature, labels):
                sess.run(optimizer, feed_dict={X: x, Y:y})

            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: feature, Y: labels})

                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

        # Evaluate it.
        W_ = sess.run(W)
        b_ = sess.run(b)

        training_cost = sess.run(
            cost,
            feed_dict={X: feature, Y: labels},
        )

        predictions = W_ * feature + b_
        reality = labels

        accuracy = helpers.regression_acc(predictions, reality)

        print 'Finished!'
        print 'Training cost=', training_cost, 'W=', W_, 'b=', b_, '\n'

        results = {
            'train': {
                'summary': {
                    'cost': training_cost,
                    'acc': accuracy,
                },
                'series': {
                    'predictions': predictions,
                    'reality': reality,
                },
            },
            'model': {
                'W': W_,
                'b': b_,
            }
        }

        return results


def plot_fit(feature, labels, predictions):
        plt.plot(feature, labels, 'ro', label='Original Data')

        plt.plot(
            feature,
            predictions,
            label='Fitted Line',
        )

        plt.legend()
        plt.show()



"""
Calculates linear regression coefficients using the normal equation method. I'm hoping
we can use this as a baseline to compare other learning algorithms against (e.g. if a
simple linear regression model can't get close to the normal equation solution,
something is wrong with it.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import ml.defaults as defaults
import ml.models.helpers as helpers


def add_bias_column(data):
    """
    Our other models use separate weight and bias matrices, but the normal equation
    uses only a single matric, so in order to include a bias in the predictions we have
    to add an extra feature to the data which is 1 for all examples.
    """
    new_data = np.ones((data.shape[0], data.shape[1] + 1))
    new_data[:,:-1] = data

    return new_data


def normal_equation_definition(x, y):
    x_t = tf.transpose(x)

    norm_eqn = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(x_t, x)), x_t), y)

    return norm_eqn


def build_train_test_eval(training_data, training_labels, test_data, test_labels):
    # Reshape the input data.
    training_data = add_bias_column(training_data)
    test_data = add_bias_column(test_data)

    # Set up the model.
    feature_count = training_data.shape[1]
    num_samples = training_data.shape[0]

    x = tf.placeholder(tf.float32, [None, feature_count])
    y = tf.placeholder(tf.float32, [None, 1])

    norm_eqn = normal_equation_definition(x, y)

    W = tf.placeholder(tf.float32, [feature_count, 1])

    pred = tf.matmul(x, W)

    cost = tf.reduce_sum(tf.pow(pred - y, 2) / (2 * num_samples))

    init = tf.initialize_all_variables()

    # Train the model.
    with tf.Session() as sess:
        sess.run(init)

        print 'Calculation solution.'

        W_ = sess.run(norm_eqn, feed_dict={x: training_data, y: training_labels})

        print 'Finished!'

        # Evaluate the model.
        cost_train = sess.run(
            cost,
            feed_dict={x: training_data, y: training_labels, W: W_},
        )

        predictions_train = sess.run(
            pred,
            feed_dict={x: training_data, y: training_labels, W: W_},
        )

        predictions_train = pd.Series([p[0] for p in predictions_train])
        reality_train = pd.Series([t[0] for t in training_labels])

        accuracy_train = helpers.regression_acc(predictions_train, reality_train)

        cost_test = sess.run(
            cost,
            feed_dict={x: test_data, y: test_labels, W: W_},
        )

        predictions_test = sess.run(
            pred,
            feed_dict={x: test_data, y: test_labels, W: W_},
        )

        predictions_test = pd.Series([p[0] for p in predictions_test])
        reality_test = pd.Series([t[0] for t in test_labels])

        accuracy_test = helpers.regression_acc(predictions_test, reality_test)

        # Output some results.
        results = {
            'train': {
                'summary': {
                    'accuracy': accuracy_train,
                    'cost': cost_train,
                }
                'series': {
                    'predictions': predictions_train,
                    'reality': reality_train,
                }
            },
            'test': {
                'summary': {
                    'accuracy': accuracy_test,
                    'cost': cost_test,
                }
                'series': {
                    'predictions': predictions_test,
                    'reality': reality_test,
                }
            },
            'model': {
                'weights': W_,
            }
        }

        return results


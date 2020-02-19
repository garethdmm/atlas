import numpy as np
import tensorflow as tf
import pandas as pd

import ml.defaults as defaults
import ml.models.helpers as helpers


def build_train_test_eval(training_data, training_labels, test_data, test_labels, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):

    # Set up the model.
    feature_count = training_data.shape[1]
    num_samples = training_data.shape[0]

    x = tf.placeholder(tf.float32, [None, feature_count])
    y = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([feature_count, 1]))
    b = tf.Variable(0.0)

    pred = tf.add(tf.matmul(x, W), b)

    cost = tf.reduce_sum(tf.pow(pred - y, 2) / (2 * num_samples))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    # Train the model.
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            for (item_x, item_y) in zip(training_data, training_labels):
                sess.run(optimizer, feed_dict={x: [item_x], y: [item_y]})

            if (epoch + 1) % display_step == 0:
                cost_ = sess.run(cost, feed_dict={x: training_data, y: training_labels})

                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_))

        print 'Finished!'

        # Evaluate the model.
        W_ = sess.run(W)
        b_ = sess.run(b)

        cost_train = sess.run(
            cost,
            feed_dict={x: training_data, y: training_labels},
        )

        predictions_train = sess.run(
            pred,
            feed_dict={x: training_data, y: training_labels},
        )

        predictions_train = pd.Series([p[0] for p in predictions_train])
        reality_train = pd.Series([t[0] for t in training_labels])

        accuracy_train = helpers.regression_acc(predictions_train, reality_train)

        cost_test = sess.run(
            cost,
            feed_dict={x: test_data, y: test_labels},
        )

        predictions_test = sess.run(
            pred,
            feed_dict={x: test_data, y: test_labels},
        )

        predictions_test = pd.Series([p[0] for p in predictions_test])
        reality_test = pd.Series([t[0] for t in test_labels])

        accuracy_test = helpers.regression_acc(predictions_test, reality_test)

        # Output some results.
        results = {
            'train': {
                'summary': {
                    'cost': cost_train,
                    'acc': accuracy_train,
                },
                'series': {
                    'predictions': predictions_train,
                    'reality': reality_train,
                },
            },
            'test': {
                'summary': {
                    'cost': cost_test,
                    'acc': accuracy_test,
                },
                'series': {
                    'predictions': predictions_test,
                    'reality': reality_test,
                },
            },
            'model': {
                'weights': W_,
                'bias': b_
            },
        }

        return results


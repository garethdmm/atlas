import numpy as np
import tensorflow as tf
import pandas as pd

import ml.defaults as defaults
import ml.models.helpers as helpers


def build_train_test_eval(feature1, feature2, labels, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):
    num_samples = feature1.shape[0]

    # Set up the model.
    X1 = tf.placeholder('float')
    X2 = tf.placeholder('float')
    Y = tf.placeholder('float')

    W1 = tf.Variable(np.random.randn(), name='w1')
    W2 = tf.Variable(np.random.randn(), name='w2')
    b = tf.Variable(np.random.randn(), name='bias')

    pred = tf.add(
        tf.add(tf.mul(X1, W1), tf.mul(X2, W2)),
        b,
    )

    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * num_samples)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    # Train it.
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            for (x1, x2, y) in zip(feature1, feature2, labels):
                sess.run(optimizer, feed_dict={X1: x1, X2: x2, Y:y})

            if (epoch + 1) % display_step == 0:
                c = sess.run(
                    cost,
                    feed_dict={X1: feature1, X2: feature2, Y: labels},
                )

                print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b)

        # Evaluate it.
        W1_ = sess.run(W1)
        W2_ = sess.run(W2)
        b_ = sess.run(W2)

        training_cost = sess.run(
            cost,
            feed_dict={X1: feature1, X2: feature2, Y: labels},
        )

        predictions = sess.run(
            pred,
            feed_dict={X1: feature1, X2: feature2, Y: labels},
        )

        predictions = pd.Series(predictions, index=labels.index)

        accuracy = helpers.regression_acc(predictions, reality)

        print 'Finished!'
        print 'Training cost=', training_cost, 'W1=', W1_, 'W2=', W2_, 'b=', b_, '\n'

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
            'test': {}
            'model': {
                'W1': W1_,
                'W2': W2_,
                'b': b_,
            }
        }

        return results


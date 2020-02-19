import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ml.defaults as defaults
import ml.models.helpers as helpers
import ml.models.model

class LogisticRegressionModel(ml.models.model.BinaryClassifierModel):
    @classmethod
    def build_train_test_eval(cls, training_data, training_labels, test_data, test_labels, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):

        feature_count = training_data.shape[1]
        classes = training_labels.shape[1]
        num_samples = training_data.shape[0]

        x = tf.placeholder(tf.float32, [None, feature_count])
        y = tf.placeholder(tf.float32, [None, classes])

        W = tf.Variable(tf.zeros([feature_count, classes]))
        b = tf.Variable(tf.zeros([classes]))

        pred = tf.nn.softmax(tf.matmul(x, W) + b)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        init = tf.initialize_all_variables()

        costs = []

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(training_epochs):
                average_cost = 0.0

                for (item_x, item_y) in zip(training_data, training_labels):
                    _, c = sess.run([optimizer, cost], feed_dict={x: [item_x], y: [item_y]})

                    average_cost += (c / len(training_data))

                if (epoch + 1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(average_cost))

            print 'Finished!'

            # Freeze the model.
            W_ = sess.run(W)
            b_ = sess.run(b)

            # Get the Predictions.
            predictions = tf.argmax(pred, 1)
            probabilities = pred
            reality = tf.argmax(y, 1)

            # Get the train results
            predictions_train, probabilities_train, reality_train = sess.run(
                [predictions, probabilities, reality],
                feed_dict={
                    x: training_data,
                    y: training_labels,
                },
            )

            acc_train, tpr_train, tnr_train, ppv_train, npv_train = cls.summary_stats(
                predictions_train,
                reality_train,
            )

            # Get the test results
            predictions_test, probabilities_test, reality_test = sess.run(
                [predictions, probabilities, reality],
                feed_dict={
                    x: test_data,
                    y: test_labels,
                },
            )

            acc_test, tpr_test, tnr_test, ppv_test, npv_test = cls.summary_stats(
                predictions_test,
                reality_test,
            )

            results = {
                'train': {
                    'summary': {
                        'acc': acc_train,
                        'tpr': tpr_train,
                        'tnr': tnr_train,
                        'ppv': ppv_train,
                        'npv': npv_train,
                    },
                    'series': {
                        'predictions': predictions_train,
                        'probabilities': probabilities_train,
                        'reality': reality_train,
                    },
                },
                'test': {
                    'summary': {
                        'acc': acc_train,
                        'tpr': tpr_test,
                        'tnr': tnr_test,
                        'ppv': ppv_test,
                        'npv': npv_test,
                    },
                    'series': {
                        'predictions': predictions_test,
                        'probabilities': probabilities_test,
                        'reality': reality_test,
                    },
                },
                'model': {
                    'W': W_,
                    'b': b_,
                },
            }

            return results



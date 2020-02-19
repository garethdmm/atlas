import tensorflow as tf
import numpy as np

import ml.defaults as defaults
import ml.models.helpers as helpers

DEFAULT_HIDDEN_LAYERS = [4, 4]  # Default to two hidden layers.

def multilayer_perceptron(x, weights, biases):
    assert (len(weights) == len(biases))

    layer_1 = tf.add(tf.matmul(x, weights[1]), biases[1])
    layer_1 = tf.nn.relu(layer_1)

    layers = [layer_1]

    for i in range(2, len(weights) + 1):
        layer = tf.add(tf.matmul(layers[-1], weights[i]), biases[i])
        layer = tf.nn.relu(layer)

        layers.append(layer)

    return layers[-1]


def create_weights_and_biases(layer_ns):
    weights = {}
    biases = {}

    assert (len(layer_ns) >= 2)

    for i in range(0, len(layer_ns) - 1):
        weights[i + 1] = tf.Variable(tf.random_normal([layer_ns[i], layer_ns[i + 1]]))
        biases[i + 1] = tf.Variable(tf.random_normal([layer_ns[i + 1]]))

    return weights, biases


def build_train_test_eval(training_data, training_labels, test_data, test_labels, hidden_layers=DEFAULT_HIDDEN_LAYERS, reg=0.5, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):

    n_input = training_data.shape[1]  # Input nodes
    n_classes = training_labels.shape[1]  # Number of classes
    layers = [n_input] + hidden_layers + [n_classes]

    # Create the inputs/outputs
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])

    # Create the weights/biases
    weights, biases = create_weights_and_biases(layers)

    pred = multilayer_perceptron(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    # TODO: add regularization back in here

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            average_cost = 0.0

            for (item_x, item_y) in zip(training_data, training_labels):
                _, c = sess.run([optimizer, cost], feed_dict={x: [item_x], y: [item_y]})

                average_cost += (c / len(training_data))

            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(average_cost))

                #costs.append(average_cost)

        print 'Finished!'

        weights_ = sess.run(weights)
        biases_ = sess.run(biases)

        # Just a way to get the results
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

        accuracy_train = helpers.acc(predictions_train, reality_train)
        tpr_train = helpers.tpr(predictions_train, reality_train)
        tnr_train = helpers.tnr(predictions_train, reality_train)
        ppv_train = helpers.ppv(predictions_train, reality_train)
        npv_train = helpers.npv(predictions_train, reality_train)

        # Get the test results
        predictions_test, probabilities_test, reality_test = sess.run(
            [predictions, probabilities, reality],
            feed_dict={
                x: test_data,
                y: test_labels,
            },
        )

        accuracy_test = helpers.acc(predictions_test, reality_test)
        tpr_test = helpers.tpr(predictions_test, reality_test)
        tnr_test = helpers.tnr(predictions_test, reality_test)
        ppv_test = helpers.ppv(predictions_test, reality_test)
        npv_test = helpers.npv(predictions_test, reality_test)

        results = {
            'train': {
                'summary': {
                    'acc': accuracy_train,
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
                    'acc': accuracy_test,
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
                'weights': weights_,
                'biases': biases_,
            }
        }

        return results



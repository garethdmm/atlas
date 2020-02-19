"""
MLP for Regression using tensorflow's built-in models. This seems to work better and
is much faster to train than our custom version. That may be because it uses more
complex optimization algorithms, or it may be doing more things behind the scenes that
I don't understand.
"""

import numpy as np
import tensorflow as tf

import ml.models.regressor_model


class TFLDNNRegressor(ml.models.regressor_model.TFLRegressorModel):
    def __init__(self, n_input, hidden_layers, l1_reg=0.0, l2_reg=0.0, dropout=None, model_dir=None, summary_steps=None, checkpoint_secs=None):
        super(TFLDNNRegressor, self).__init__()

        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.summary_steps = summary_steps or self.DEFAULT_SUMMARY_STEPS
        self.checkpoint_secs = checkpoint_secs or self.DEFAULT_CHECKPOINT_SECS

        if model_dir is None:
            model_dir = self.create_working_directory()

        self.model_dir = model_dir

        self.estimator = self.build_model()

        self.set_as_current_model()

    def build_model(self):
        feature_columns = [
            tf.contrib.layers.real_valued_column('%s' % i)
            for i in range(0, self.n_input)
        ]

        estimator = tf.contrib.learn.DNNRegressor(
            feature_columns=feature_columns,
            hidden_units=self.hidden_layers,
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(
                save_summary_steps=self.summary_steps,
                save_checkpoints_secs=self.checkpoint_secs,
            ),
            dropout=self.dropout,
        )

        return estimator

    def get_model_params(self):
        return {
            'n_input': self.n_input,
            'hidden_layers': self.hidden_layers,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'dropout': self.dropout,
        }

    @classmethod
    def load_model_from_saved_params(cls, params, model_dir):
        return cls(
            params['n_input'],
            params['hidden_layers'],
            params['l1_reg'],
            params['l2_reg'],
            params['dropout'] if 'dropout' in params else None,
            model_dir=model_dir,
        )

    def number_of_parameters(self):
        return sum([x.size for x in self.estimator.weights_ + self.estimator.bias_])

    def get_activations(self, features):
         """
         Returns an array corresponding to the node activations of the network on the
         input features.
         """

         activations = [features]

         for layer_number, num_layer_nodes in enumerate(self.hidden_layers):
             layer_activations = []
             last_layer_activations = activations[-1]

             for node_number in range(0, num_layer_nodes):
                 weights = self.estimator.weights_[layer_number][node_number]
                 biases = self.estimator.bias_[layer_number][node_number]

                 node_input = np.dot(last_layer_activations, weights) + biases
                 node_activation = max(node_input, 0)

                 layer_activations.append(node_activation)

             activations.append(np.array(layer_activations))

         return np.array(activations[1:]).tolist()

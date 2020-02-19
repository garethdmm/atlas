"""
Linear regressor using the built-in models in tf.contrib.learn.
"""

import tensorflow as tf

import ml.models.regressor_model


class TFLLinearRegressor(ml.models.regressor_model.TFLRegressorModel):
    validation_metrics = {}  # These are broken on this model right now. Unsure why.

    def __init__(self, n_input, model_dir=None, summary_steps=None, checkpoint_secs=None):
        super(TFLLinearRegressor, self).__init__()

        self.n_input = n_input
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

        estimator = tf.contrib.learn.LinearRegressor(
            feature_columns=feature_columns,
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(
                save_summary_steps=self.summary_steps,
                save_checkpoints_secs=self.checkpoint_secs,
            )
        )

        return estimator

    def get_model_params(self):
        return {
            'n_input': self.n_input,
        }

    @classmethod
    def load_model_from_saved_params(cls, params, model_dir):
        return cls(
            params['n_input'],
            model_dir=model_dir,
        )

    def number_of_parameters(self):
        return len(self.estimator.weights_) + len(self.estimator.bias_)


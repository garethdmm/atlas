"""
This subclass of FeatureLabelSet allows us to get aligned training/validation sets for
multiple different featuresets at once, which is important for validating ensemble
models.
"""
import pandas as pd

import ml.defaults as defaults
import ml.data.feature_label_set


class EnsembleFeatureLabelSet(ml.data.feature_label_set.FeatureLabelSet):
    def __init__(self, featuresets):
        self.featuresets = featuresets
        self.labels = featuresets[0].labels
        self.resolution = featuresets[0].features[0].pandas_freq()

        # TODO: Add in asserts to make sure we're not doing bad things.

    def get_formatted_data(self, *args, **kwargs):
        # Get all the dataframes from the featuresets.
        dfs = [f.get_formatted_data(*args, **kwargs) for f in self.featuresets]

        # Align the dataframes.
        max_dfs = pd.concat(dfs, axis=1)
        max_dfs = self.drop_nan_rows(max_dfs)

        idx = max_dfs.index

        reindexed_dfs = []

        for df in dfs:
            reindexed_dfs.append(df.reindex(idx))

        return reindexed_dfs

    def get_data_for_train_test_run(self, db, train_start=defaults.TRAIN_START, train_end=defaults.TRAIN_END, test_start=defaults.TEST_START, test_end=defaults.TEST_END, for_classification=True, interpolate=False, max_interpolate=None):
        training_data_dfs = self.get_formatted_data(
            db,
            train_start,
            train_end,
            realign=True,
            rescale=True,
            for_classification=for_classification,
            interpolate=interpolate,
            max_interpolate=max_interpolate,
            all_valid=True,
        )

        test_data_dfs = self.get_formatted_data(
            db,
            test_start,
            test_end,
            realign=True,
            rescale=True,
            for_classification=for_classification,
            interpolate=interpolate,
            max_interpolate=max_interpolate,
            all_valid=True,
        )

        # TODO: I think we can cut out this code by moving it into a "prepare-data"-ish
        # function in the superclass.
        training_datas = [df.iloc[:, 0:-1] for df in training_data_dfs]
        training_labels = training_data_dfs[0].iloc[:,-1]

        test_datas = [df.iloc[:, 0:-1] for df in test_data_dfs]
        test_labels = test_data_dfs[0].iloc[:, -1]

        self.print_retrieved_data_summary(
            train_start,
            train_end,
            test_start,
            test_end,
            training_datas[0],
            test_datas[0],
        )

        return training_datas, training_labels, test_datas, test_labels


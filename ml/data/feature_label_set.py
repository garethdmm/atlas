"""
A class we use to group features and labels together so that it is easy to retrieve
data and do processing on the whole set at once through a simple interface (get_data/
get_formatted_data).
"""
import imblearn.under_sampling
import matplotlib.pyplot as plt

import ml.data.conversion as conversion
from ml.data.feature import *
import ml.data.helpers as data_helpers
import ml.data.rebalance
import ml.defaults as defaults


class FeatureLabelSet(object):
    def __init__(self, features, labels):
        assert len(set([f.resolution for f in features + labels])) == 1

        self.features = features
        self.labels = labels
        self.resolution = features[0].pandas_freq()

    def _get_raw_data(self, db, start, end, interpolate=False, max_interpolate=None):
        feature_data = []
        label_data = []
        localcache = {}

        for feature in self.features:
            feature_data.append(feature.get_data(
                db,
                start,
                end,
                interpolate=interpolate,
                max_interpolate=max_interpolate,
                cache=None,
            ))

        for label in self.labels:
            label_data.append(label.get_data(
                db,
                start,
                end,
                interpolate=interpolate,
                max_interpolate=max_interpolate,
                cache=None,
            ))

        return feature_data, label_data

    def get_formatted_data(self, db, start=defaults.TRAIN_START, end=defaults.TEST_END, realign=False, rescale=False, for_classification=False, interpolate=False, max_interpolate=None, all_valid=False, rescale_labels=False, percentiles=None, handle_boundary=None):
        feature_data, label_data = self._get_raw_data(
            db,
            start,
            end,
            interpolate=interpolate,
            max_interpolate=max_interpolate,
        )

        if realign is True:
            feature_data, label_data = self.realign_data(feature_data, label_data)

        rescale_params = None

        if rescale is True:
            feature_data, label_data, rescale_params = self.rescale_data(
                feature_data,
                label_data,
                rescale_labels=rescale_labels,
            )

        if for_classification is True:
            if percentiles is None:
                label_data = [
                    conversion.decimal_labels_to_binary_categorical(
                        label_data[0],
                        handle_boundary=handle_boundary,
                    )
                ]
            else:
                label_data = [
                    conversion.decimal_labels_to_categorical_by_percentile(
                        label_data[0],
                        percentiles,
                        handle_boundary=handle_boundary,
                    )
                ]

        data = pd.concat(feature_data + label_data, axis=1)

        if all_valid is True:
            data = self.drop_nan_rows(data)

        return data, rescale_params

    def drop_nan_rows(self, data):
        start_length = len(data)

        data = data.dropna()

        final_length = len(data)

        dropped_rows_as_percent = (start_length - final_length) / float(start_length)

        if dropped_rows_as_percent > 0.0:
            logger.info('Dropped %.2f%% of %s rows due to nan values' % (
                dropped_rows_as_percent,
                len(data),
            ))

        return data

    def target_number_of_rows(self, start, end):
        idx = pd.date_range(
            start=start,
            end=end,
            freq=self.resolution,
        )

        return len(idx)

    def get_data_for_train_test_run(self, db, train_start=defaults.TRAIN_START, train_end=defaults.TRAIN_END, test_start=defaults.TEST_START, test_end=defaults.TEST_END, for_classification=True, interpolate=False, max_interpolate=None, rebalance=False, percentiles=None, rebalance_method=None, handle_boundary=None):
        data_df, rescale_params = self.get_formatted_data(
            db,
            train_start,
            test_end,
            realign=True,
            rescale=True,
            for_classification=for_classification,
            interpolate=interpolate,
            max_interpolate=max_interpolate,
            all_valid=True,
            percentiles=percentiles,
            handle_boundary=handle_boundary,
        )

        # Might be throwing out slightly more data here than we need to be with these
        # closures.
        training_data_df = data_df[
            (data_df.index > train_start) & (data_df.index <= train_end)
        ]

        test_data_df = data_df[
            (data_df.index > test_start) & (data_df.index <= test_end)
        ]

        training_data = training_data_df.iloc[:, 0:-1]
        training_labels = training_data_df.iloc[:, -1]

        test_data = test_data_df.iloc[:, 0:-1]
        test_labels = test_data_df.iloc[:, -1]

        self.print_retrieved_data_summary(
            train_start,
            train_end,
            test_start,
            test_end,
            training_data,
            test_data,
        )

        if rebalance is True:
            training_data, training_labels = self.rebalance_data(
                training_data,
                training_labels,
                rebalance_method,
            )

        return training_data, training_labels, test_data, test_labels, rescale_params

    def rebalance_data(self, feature_data, label_data, rebalance_method=None):
        """
        Rebalance a classification dataset by the given method. Dispatches calls to the
        rebalance library.
        """

        # Weird default argument behaviour in this function. We'll clean it up soon.
        if rebalance_method is None or rebalance_method == 'undersample':
            features_balanced, labels_balanced = ml.data.rebalance.undersample(
                feature_data,
                label_data,
            )
        elif rebalance_method == 'oversample':
            features_balanced, labels_balanced = ml.data.rebalance.oversample(
                feature_data,
                label_data,
            )
        elif rebalance_method == 'adasyn':
            features_balanced, labels_balanced = ml.data.rebalance.adasyn(
                feature_data,
                label_data,
            )
        elif rebalance_method == 'smote':
            features_balanced, labels_balanced = ml.data.rebalance.smote(
                feature_data,
                label_data,
            )
        else:
            raise NotImplementedError('Unknown rebalance method %s' % rebalance_method)

        return features_balanced, labels_balanced

    def realign_data(self, feature_data, label_data):
        """
        Take a list of pandas series' that are valid across different subranges of a
        higher range and realign them to the minimum index between them.

        Add in an assert or two to make sure that there are no holes inside the final
        range.
        """

        realigned_features = []
        realigned_labels = []

        all_data = feature_data + label_data

        last_start = max([d.first_valid_index() for d in all_data])
        first_end = min([d.last_valid_index() for d in all_data])

        idx = pd.date_range(
            start=last_start,
            end=first_end,
            freq=self.resolution,
        )

        realigned_features = [d.reindex(idx) for d in feature_data]
        realigned_labels = [d.reindex(idx) for d in label_data]

        return realigned_features, realigned_labels

    def rescale_data(self, feature_data, label_data, rescale_labels=False):
        rescale_params = []
        rescaled_features = []
        rescaled_labels = []

        for feature in feature_data:
            rescaled, mean, std = processing.rescale_feature(feature)

            rescaled_features.append(rescaled)
            rescale_params.append({
                'mean': mean,
                'std': std,
            })

        if rescale_labels is True:
            for label in label_data:
                rescaled, mean, std = processing.rescale_feature(label)

                rescaled_labels.append(rescaled)
                rescale_params.append({
                    'mean': mean,
                    'std': std,
                })
        else:
            rescaled_labels = label_data

        return rescaled_features, rescaled_labels, rescale_params

    def plot_data(self, start, end, subplots=True):
        feature_data, label_data = self.get_data(start, end)
        feature_data, label_data = self.realign_data(feature_data, label_data)
        feature_data, label_data = self.rescale_data(feature_data, label_data)

        plt.ion()

        if subplots is True:
            pd.concat(feature_data + label_data, axis=1).plot(subplots=True)

            for ax in plt.figure(1).axes:
                ax.legend(fontsize=10)
        else:
            ax = plt.subplot(211)
            pd.concat(feature_data, axis=1).plot(
                title='Features',
                subplots=False,
                ax=ax,
            )
            plt.legend(fontsize=10)

            ax = plt.subplot(212)
            plt.legend(fontsize=10)
            pd.concat(label_data, axis=1).plot(title='Label', subplots=False, ax=ax)
            plt.legend(fontsize=10)

    def print_retrieved_data_summary(self, train_start, train_end, test_start, test_end, training_data, test_data):
        target_rows_train = self.target_number_of_rows(train_start, train_end)
        target_rows_test = self.target_number_of_rows(test_start, test_end)
        real_rows_train = len(training_data)
        real_rows_test = len(test_data)
        coverage_train = 100 * real_rows_train / target_rows_train
        coverage_test = 100 * real_rows_test / target_rows_test

        logger.info('\n')
        logger.info('Dataset results (before rebalance):')
        logger.info('\tTrain: Got %s of %s possible rows (%s%%)' % (
            real_rows_train,
            target_rows_train,
            coverage_train,
        ))
        logger.info('\tTest:  Got %s of %s possible rows (%s%%)' % (
            real_rows_test,
            target_rows_test,
            coverage_test,
        ))

    def print_coverages_over_range(self, db, run_start=defaults.TRAIN_START, run_end=defaults.TEST_END):
        """
        Helper function to look at how complete each feature in this set is over a
        given range, for example the range that we want to do a train/validate run on.
        """

        print '\nFeature coverages over range %s - %s\n' % (
            str(run_start),
            str(run_end),
        )

        print 'Series'.ljust(60) + 'Coverage'

        for f in self.features:
            run_coverage = f.coverage_over_range(db, run_start, run_end)
            name = f._get_pretty_name()

            print name.ljust(60) + ('%.4f' % run_coverage)

    def print_valid_ranges_and_coverage(self, db):
        """
        Helper function to look at how where our data starts/ends and how complete it is
        for each feature in the set.
        """

        print '\nFeature maximum valid ranges and coverages over that range\n'
        print '%s%s%s%s' % (
            'Series'.ljust(60),
            'Start'.ljust(24),
            'End'.ljust(24),
            'Coverage',
        )

        for f in self.features:
            valid_start = f.first_valid_timestamp(db)
            valid_end = f.last_valid_timestamp(db)

            valid_range_coverage = f.coverage_over_valid_range(
                db,
            )

            name = f._get_pretty_name()

            print '%s%s%s%.4f' % (
                name.ljust(60),
                str(valid_start).ljust(24),
                str(valid_end).ljust(24),
                valid_range_coverage
            )

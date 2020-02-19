"""
Ensemble models. These take in a list of other models ("ensemble members"), run them
each individually on the data under consideration, and combine them to create a single
prediction.
"""

import numpy as np
import pandas as pd

import ml.models.model
import ml.models.classifier_model


class Ensemble(ml.models.model.Model):
    def __init__(self, members):
        self.members = members

    def prepare_dataset(self, features, labels):
        return features, labels

class BinaryClassifierEnsemble(Ensemble, ml.models.classifier_model.TFLBinaryClassifierModel):
    def _is_symmetric_featureset(self, features):
        """
        This function tells us if the incoming feature data is for a symmetric ensemble
        (all the same featuresets), or an asymmetric one.
        """
        return type(features) == np.ndarray

    def get_summary(self, features, labels):
        """Hack because our inheritence graph isn't perfect yet."""
        return {'loss': -1}

    def get_probabilities_from_members(self, features):
        probabilities = []

        if self._is_symmetric_featureset(features):
            for m in self.members:
                probs = m.get_probabilities_on_dataset(features)
                probabilities.append(probs)

        else:
            for i, m in enumerate(self.members):
                probs = m.get_probabilities_on_dataset(m.vectorize(features[i]))
                probabilities.append(probs)

        return probabilities


class MaxConfidenceEnsemble(BinaryClassifierEnsemble):
    """
    This ensemble model considers the probabilities each of it's ensemble members assign
    to a prediction and chooses the highest one.
    """

    def get_predictions_on_dataset(self, features):
        probabilities = self.get_probabilities_on_dataset(features)
        predictions = np.array(pd.DataFrame(probabilities).T.idxmax())

        return predictions

    def get_probabilities_on_dataset(self, features):
        probabilities = self.get_probabilities_from_members(features)
        return self._get_probabilities_from_ensemble_members(probabilities)

    def _get_probabilities_from_ensemble_members(self, ensemble_probs):
        """
        Takes in the probabilities list of each of the ensemble members (a list of lists
        of duples),
        """

        # Figure out whether we're taking the probabilities from the first or second
        # member.
        prob_columns = []
        
        for p in ensemble_probs:
            prob_columns.append(p[:,0])
            prob_columns.append(p[:,1])

        probs_columns = [pd.Series(p) for p in prob_columns]
        probs_df = pd.concat(probs_columns, axis=1)
        which_one = (probs_df.T.idxmax() / 2).astype(int)  # Uses Int as rounding.

        # Iterate through the ensemble probabilities and at each step take the one the
        # above operations indicated contained the highest probability.
        best_probs = []

        for i, which in enumerate(which_one):
            best_probs.append(ensemble_probs[which][i])

        return np.array(best_probs)

        
class MajorityVoteEnsemble(BinaryClassifierEnsemble):
    """
    This ensemble model takes a vote from each of it's members and predicts the majority
    opinion. The current implementation defaults to negative in case of a tie, but this
    could be changed.
    """

    def get_predictions_on_dataset(self, features):
        probabilities = self.get_probabilities_from_members(features)
        return self._get_predictions_from_ensemble_members(probabilities)

    def get_probabilities_on_dataset(self, features):
        """
        Currently unsure how to define this. Average of the confidence of the dominant
        predictions?
        """
        return np.array([])

    def _get_predictions_from_ensemble_members(self, ensemble_probs):
        """
        This should probably take in a dataframe rather than a list of lists.

        np.argmax appears to return the first index of the highest value in the event of
        a tie. That means that this ensemble will default to predicting negative if a
        vote is tied. Same logic as "predict positive iff
        |positives votes| > 0.5 * num_ensembles else predict negative"

        Will have to be careful about this behaviour. Could be avoided by never using
        even-sized ensembles.
        """
        predictions = []

        for probs in ensemble_probs:
            predictions.append([np.argmax(p) for p in probs])

        predictions = np.array(predictions).T

        votes_per_index = [np.bincount(v) for v in predictions]
        df = pd.DataFrame(votes_per_index)

        return np.array(df.T.idxmax())


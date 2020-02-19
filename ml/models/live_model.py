"""
A wrapper around models which packages them so they can be run in the model runner.
"""
from datetime import datetime
import delorean

import importlib
import numpy as np
import sqlalchemy
import json
import pandas as pd

import ml.models.regressor_model
from gryphon.lib.exchange.base import Exchange
from gryphon.lib.exchange.consts import Consts
from gryphon.lib.logger import get_logger
from gryphon.lib.models.datum import Datum, DatumRecorder
from gryphon.lib.money import Money

logger = get_logger(__name__)


class LiveModel(ml.models.regressor_model.TFLRegressorModel):
    """
    live_model = LiveModel.load_model('dnnr_strength_bitstamp_10m_test')

    Current only supports models that are
    - regression
    - featureset is 1,5,10 strenghts on both sides
    - target is future 1-step-forward usd midpoint change in cents

    These models keep track of the performance of their predictions by saving each
    prediction along with the most recent realized state in datums. When a model is
    loaded, it loads it's past predictions into it's memory and adds new predictions on
    to this pandas series.
    """

    def __init__(self, model_name, gryphon_session):
        self.model_name = model_name
        self.model = self.load_model(model_name)
        self.rescale_params = self._load_rescale_params(model_name)
        self.gryphon_session = gryphon_session

        self._load_past_predictions()
        self._load_past_realities()

    def tick(self, timestamp, orderbook):
        feature_data = self._get_features_from_ob(orderbook)
        prediction = self.model.get_predictions_on_dataset(feature_data)[0]
        formatted_output = self._format_output(prediction)

        current_target_value = self._get_current_value_of_target(orderbook)

        activations = []

        if hasattr(self.model, 'get_activations'):
            activations = self.model.get_activations(feature_data[0])

        self._record_stats(timestamp, prediction, current_target_value, activations)

        # self._log()

        return formatted_output

    def train(*args, **kwargs):
        raise NotImplementedError

    @property
    def stat_summary_datum_type(self):
        return 'ML_STAT_SUMMARY_FOR_%s' % self.model_name.upper()

    @property
    def prediction_datum_type(self):
        return 'ML_PRED_FOR_%s' % self.model_name.upper()

    @property
    def reality_datum_type(self):
        return 'ML_REL_FOR_%s' % self.model_name.upper()

    def _load_past_predictions(self):
        datums = self.gryphon_session.query(Datum)\
            .filter(Datum.time_created > datetime(2017, 3, 20))\
            .filter(Datum.datum_type == self.prediction_datum_type)\
            .all()

        values = [float(d.numeric_value) for d in datums]

        timestamps = [
            delorean.parse(json.loads(d.meta_data)['timestamp']).datetime for d in datums
        ]

        self._past_predictions = pd.Series(values, timestamps)

    def _load_past_realities(self):
        datums = self.gryphon_session.query(Datum)\
            .filter(Datum.time_created > datetime(2017, 3, 20))\
            .filter(Datum.datum_type == self.reality_datum_type)\
            .all()

        values = [float(d.numeric_value) for d in datums]

        timestamps = [
            delorean.parse(json.loads(d.meta_data)['timestamp']).datetime
            for d in datums
        ]

        self._past_realities = pd.Series(values, timestamps)

    def _load_rescale_params(self, model_name):
        model_dir = self.model_dir_for_name(model_name)
        metadata = self.load_metadata(model_dir)

        rescale_params = metadata['rescale_params']

        return rescale_params

    def _format_output(self, predictions):
        return predictions

    def _get_current_value_of_target(self, orderbook):
        return self._get_midpoint(orderbook)

    def _get_midpoint(self, orderbook):
        bid_quote = Exchange.price_quote_from_orderbook(
            orderbook,
            'BID',
            Money(20, 'BTC'),
        )

        bid_price = float(bid_quote['price_for_order'])

        ask_quote = Exchange.price_quote_from_orderbook(
            orderbook,
            'ASK',
            Money(20, 'BTC'),
        )

        ask_price = float(ask_quote['price_for_order'])

        return (bid_price + ask_price) / 2

    def _get_features_from_ob(self, orderbook):
        levels = [
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.BID, 1)),
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.BID, 5)),
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.BID, 10)),
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.ASK, 1)),
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.ASK, 5)),
            float(Exchange.orderbook_strength_at_slippage(orderbook, Consts.ASK, 10)),
        ]

        levels = self._rescale_features(levels)

        feature_data = np.array([levels])

        return feature_data

    def _rescale_features(self, features):
        rescaled_features = []

        for rescale, feature in zip(self.rescale_params, features):
            feature = (feature - rescale['mean']) / rescale['std']

            rescaled_features.append(feature)

        return rescaled_features

    def _set_in_redis(self, r):
        """
        Future use. Any active live model should be able to set it's prediction state in
        redis for gryphon-fury bots to pick up.
        """
        pass

    def _log(self):
        logger.info(self._past_predictions)
        logger.info(self._past_realities)

    def _record_stats(self, timestamp, prediction, current_value, activations):
        """
        This function could definitely use some cleanup.
        """

        DatumRecorder().record(
            self.prediction_datum_type,
            numeric_value=prediction,
            meta_data={
                'timestamp': timestamp.isoformat(),
            },
        )

        DatumRecorder().record(
            self.reality_datum_type,
            numeric_value=current_value,
            meta_data={
                'timestamp': timestamp.isoformat(),
            },
        )

        s1 = pd.Series([prediction], index=[timestamp])
        s2 = pd.Series([current_value], index=[timestamp])

        self._past_predictions = self._past_predictions.append(s1)
        self._past_realities = self._past_realities.append(s2)

        # Calculate ACC, NPV, PPV, Loss on current predictions/realities and save
        diffs = self._past_realities.drop_duplicates().diff()

        df = pd.concat([self._past_predictions.drop_duplicates(), diffs], axis=1)
        df = df.dropna()

        if df.count()[0] > 0:
            acc, tpr, tnr, ppv, npv = self.summary_stats(df.iloc[:, 0], df.iloc[:, 1])

            DatumRecorder().record(
                self.stat_summary_datum_type,
                numeric_value=0,
                meta_data={
                    'timestamp': timestamp.isoformat(),
                    'acc': acc if not np.isnan(acc) else 0,
                    'ppv': ppv if not np.isnan(ppv) else 0,
                    'npv': npv if not np.isnan(npv) else 0,
                    'activations': activations,
                },
            )
        else:
            DatumRecorder().record(
                self.stat_summary_datum_type,
                numeric_value=0,
                meta_data={
                    'timestamp': timestamp.isoformat(),
                    'acc': 0,
                    'ppv': 0,
                    'npv': 0,
                    'activations': activations,
                },
            )

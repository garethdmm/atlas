import pyximport; pyximport.install()
import gryphon.lib; gryphon.lib.prepare()

from datetime import datetime, timedelta
import os
import pickle

from cdecimal import Decimal
from delorean import Delorean, parse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from gryphon.lib.exchange.base import Exchange
from gryphon.lib.exchange.bitstamp_btc_usd import BitstampBTCUSDExchange
from gryphon.lib.exchange.gemini_btc_usd import GeminiBTCUSDExchange
from gryphon.lib.logger import get_logger
from gryphon.lib.models.atlaszero.metric import Metric
from gryphon.lib.models.atlaszero import metric_types
from gryphon.lib.models.datum import DatumRecorder
from gryphon.lib.models.emeraldhavoc.orderbook import Orderbook
from gryphon.lib.money import Money
from gryphon.lib import session

from ml.defaults import *
import ml.data.feature as feature
from ml.data.feature_label_set import FeatureLabelSet
import ml.data.preconfigured_feature_label_sets as featuresets
from ml.models.ensemble import MaxConfidenceEnsemble
import ml.visualize as v
from ml.models.live_model import LiveModel
from ml.models.tfl_model import TFLModel
from ml.models.tfl_linear_classifier import TFLLinearClassifier
from ml.models.tfl_linear_regression import TFLLinearRegressor
from ml.models.tfl_dnn_classifier import TFLDNNClassifier
from ml.models.tfl_dnn_regressor import TFLDNNRegressor

logger = get_logger(__name__)

try:
    db = session.get_a_mysql_session(os.environ['ATLAS_ZERO_DB_CRED'])
    gdb = session.get_a_mysql_session(os.environ['GRYPHON_CRED'])
    eh_dbs = get_emerald_havoc_dbs()
except:
    print 'Could not get one or more database connections'

b = BitstampBTCUSDExchange()
g = GeminiBTCUSDExchange()

plt.ion()

DatumRecorder().create(db=gdb)


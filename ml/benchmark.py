import pyximport; pyximport.install()
import gryphon.lib; gryphon.lib.prepare()

from datetime import datetime
import os

from cdecimal import Decimal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analytics.tools import data_loading
from gryphon.lib.logger import get_logger
from gryphon.lib.models.atlaszero.metric import Metric
from gryphon.lib.models.atlaszero import metric_types
from gryphon.lib.models.emeraldhavoc.orderbook import Orderbook
from gryphon.lib import session


import ml.data.feature as feature
from ml.data.feature_label_set import FeatureLabelSet
import ml.data.preconfigured_feature_label_set as featuresets
import ml.models.helpers as h
from ml.models.tfl_linear_classifier import TFLLinearClassifier as linc
from ml.models.tfl_linear_regression import TFLLinearRegressor as linr
from ml.models.tfl_dnn_classifier import TFLDNNClassifier as dnnc
from ml.models.tfl_dnn_regressor import TFLDNNRegressor as dnnr

logger = get_logger(__name__)

db = session.get_a_mysql_session(os.environ['ATLAS_ZERO_DB_CRED'])
eh_db = session.get_a_mysql_session(os.environ['EH_DB_CRED'])

td, tl, tdt, tlt = featuresets.Lots(db).get_data_for_train_test_run()

start = datetime.now()

results = dnnc.build_train_test_eval(
    td,
    tl,
    tdt,
    tlt,
    hidden_layers=[14, 28, 28, 14],
    training_epochs=5000,
)

end = datetime.now()

elapsed = end - start

print 'Completed in %s seconds' % elapsed.seconds

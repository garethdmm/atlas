"""
A set of classes which that give a lovely builder interface to identify any feature we
could want to use for machine learning and abstracts all data retrieval.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy.orm
from sqlalchemy import func

import ml.data.processing as processing
from gryphon.lib.logger import get_logger
from gryphon.lib.models.atlaszero.metric import Metric
import gryphon.lib.models.atlaszero.metric_types as metric_types

logger = get_logger(__name__)


class Feature(object):
    series_type = None
    exchange_name = None
    source = None
    resolution = None
    param = None
    param_name = None

    def bitstamp(self):
        self.exchange_name = 'bitstamp'
        return self

    def bitfinex(self):
        self.exchange_name = 'bitfinex'
        return self

    def kraken(self):
        self.exchange_name = 'kraken'
        return self

    def itbit(self):
        self.exchange_name = 'itbit'
        return self

    def okcoin(self):
        self.exchange_name = 'okcoin'
        return self

    def coinbase(self):
        self.exchange_name = 'coinbase'
        return self

    def gemini(self):
        self.exchange_name = 'gemini'
        return self

    def hour(self):
        self.resolution = '1hr'
        return self

    def ten_min(self):
        self.resolution = '10m'
        return self

    def five_min(self):
        self.resolution = '5m'
        return self

    def one_min(self):
        self.resolution = '1m'
        return self

    def pandas_freq(self):
        if self.resolution == '1hr':
            return 'H'
        elif self.resolution == '10m':
            return '10T'
        elif self.resolution == '5m':
            return '5T'
        elif self.resolution == '1m':
            return '1T'

    def get_cache_key(self, start, end, series_id):
        key = '%s;%s;%s' % (series_id, start, end)
        key = key.replace(' ', '_')

        return key

    def has_duplicates(self, db):
        return len(self.duplicates(db)) != 0

    def duplicates(self, db):
        metrics_per_timestamp = db.query(Metric.timestamp, func.count(Metric))\
            .filter(Metric.metric_type == self._get_series_id())\
            .group_by(Metric.timestamp)\
            .all()

        duplicated_timestamps = [m[0] for m in metrics_per_timestamp if m[1] != 1]

        return duplicated_timestamps

    def first_valid_timestamp(self, db):
        m = db.query(Metric)\
            .filter(Metric.metric_type == self._get_series_id())\
            .filter(Metric.value != None)\
            .order_by(Metric.timestamp.asc())\
            .first()

        return m.timestamp

    def last_valid_timestamp(self, db):
        m = db.query(Metric)\
            .filter(Metric.metric_type == self._get_series_id())\
            .filter(Metric.value != None)\
            .order_by(Metric.timestamp.desc())\
            .first()

        return m.timestamp

    def get_valid_range(self, db):
        return self.first_valid_timestamp(db), self.last_valid_timestamp(db)

    def get_num_valid_points_in_range(self, db, start, end):
        num_data_points = db.query(Metric)\
            .filter(Metric.metric_type == self._get_series_id())\
            .filter(Metric.timestamp > start)\
            .filter(Metric.timestamp <= end)\
            .filter(Metric.value != None)\
            .count()

        return num_data_points

    def coverage_over_valid_range(self, db):
        valid_start, valid_end = self.get_valid_range(db)

        return self.coverage_over_range(db, valid_start, valid_end)

    def coverage_over_range(self, db, start, end):
        num_data_points = self.get_num_valid_points_in_range(db, start, end)

        idx = pd.date_range(
            start=start,
            end=end,
            freq=self.pandas_freq(),
        )

        return num_data_points / float(len(idx))

    def get_raw_data_from_db(self, db, start, end, series_id):
        data = db.query(Metric)\
            .filter(Metric.metric_type == series_id)\
            .filter(Metric.timestamp > start)\
            .filter(Metric.timestamp <= end)\
            .all()

        return data

    def get_data_with_caching(self, db, cache, start, end, series_id):
        key = self.get_cache_key(start, end, series_id)

        if cache is None:
            data = self.get_raw_data_from_db(db, start, end, series_id)

            return data

        if key in cache:
            return cache[key]
        else:
            data = self.get_raw_data_from_db(db, start, end, series_id)

            if cache is not None:
                cache[key] = data

            return data

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        data = self.get_data_with_caching(db, cache, start, end, self._get_series_id())

        data = Metric.convert_metric_series_to_pandas(data)
        data = pd.Series(data, name=self._get_pretty_name())

        idx = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq=self.pandas_freq(),
        )

        data = data.reindex(idx)

        if interpolate is True:
            data = self.interpolate_data(data, max_interpolate)

        return data

    def interpolate_data(self, data, max_interpolate=None):
        valid_data_points = data.count()

        data = data.interpolate(limit=max_interpolate) # None argument implies infinity

        # TODO: There is a bug here whereby if the series begins or ends with nan's,
        # interpolate will not fill them in and this will throw an error.
        if max_interpolate is None:
            assert data.count() == len(data)

        interpolated_points = data.count() - valid_data_points
        interpolated_points_as_percent = (interpolated_points / float(len(data))) * 100

        if interpolated_points_as_percent > 0.0:
            logger.info('Interpolated %.2f%% of %s points in %s' % (
                interpolated_points_as_percent,
                len(data),
                self._get_series_name(),
            ))

        return data

    def _get_pretty_name(self):
        return '%s %s %s %s %s' % (
            type(self).__name__,
            self.exchange_name.capitalize(),
            self.resolution,
            self.param,
            self.param_name,
        )

    def _get_series_name(self):
        series_name = None

        if self.param is not None:
            series_name = '%s-%s-%s-%s-%.2f%s' % (
                self.series_type,
                self.exchange_name,
                self.source,
                self.resolution,
                self.param,
                self.param_name,
            )
        else:
            series_name = '%s-%s-%s-%s' % (
                self.series_type,
                self.exchange_name,
                self.source,
                self.resolution,
            )

        return series_name

    def _get_series_id(self):
        return metric_types.get_metric_type_int(self._get_series_name())

    def plot(self, db, start, end, interpolate=False, max_interpolate=None):
        data = self.get_data(
            db,
            start,
            end,
            interpolate=interpolate,
            max_interpolate=max_interpolate,
        )

        plt.ion()
        data.plot()


class InterExchangeSpread(Feature):
    """
    The spread between exchange 1 and 2 as a percent. Specifically this feature is
    a number x such that e1*(x + 1) = e2.

    Be very careful you don't have long ranges with no data on either of these
    exchanges or the output will be unpredictable!
    """

    def __init__(self, feature1, feature2):
        assert feature1.resolution == feature2.resolution
        assert type(feature1) == type(feature2) == Midpoint

        self.feature1 = feature1
        self.feature2 = feature2
        self.resolution = self.feature1.resolution

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        series1 = self.feature1.get_data(db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        series2 = self.feature2.get_data(db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        new_data = (series2 / series1) - 1

        new_data = pd.Series(new_data, name=self._get_pretty_name())

        return new_data

    def _get_pretty_name(self):
        return '%s %s %s %s' % (
            type(self).__name__,
            self.feature1.exchange_name.capitalize(),
            self.feature2.exchange_name.capitalize(),
            self.resolution,
        )

    def first_valid_timestamp(self, db):
        m1 = sqlalchemy.orm.aliased(Metric)
        m2 = sqlalchemy.orm.aliased(Metric)

        first_metrics = db.query(m1, m2)\
            .join(m2, m1.timestamp == m2.timestamp)\
            .filter(m1.metric_type == self.feature1._get_series_id())\
            .filter(m2.metric_type == self.feature2._get_series_id())\
            .filter(m1.value != None)\
            .filter(m2.value != None)\
            .order_by(m1.timestamp.asc())\
            .first()

        return first_metrics[0].timestamp

    def last_valid_timestamp(self, db):
        m1 = sqlalchemy.orm.aliased(Metric)
        m2 = sqlalchemy.orm.aliased(Metric)

        last_metrics = db.query(m1, m2)\
            .join(m2, m1.timestamp == m2.timestamp)\
            .filter(m1.metric_type == self.feature1._get_series_id())\
            .filter(m2.metric_type == self.feature2._get_series_id())\
            .filter(m1.value != None)\
            .filter(m2.value != None)\
            .order_by(m1.timestamp.desc())\
            .first()

        return last_metrics[0].timestamp

    def get_num_valid_points_in_range(self, db, start, end):
        m1 = sqlalchemy.orm.aliased(Metric)
        m2 = sqlalchemy.orm.aliased(Metric)

        num_data_points = db.query(m1, m2)\
            .join(m2, m1.timestamp == m2.timestamp)\
            .filter(m1.metric_type == self.feature1._get_series_id())\
            .filter(m2.metric_type == self.feature2._get_series_id())\
            .filter(m1.timestamp > start)\
            .filter(m1.timestamp <= end)\
            .filter(m2.timestamp > start)\
            .filter(m2.timestamp <= end)\
            .filter(m1.value != None)\
            .filter(m2.value != None)\
            .count()

        return num_data_points


class Volume(Feature):
    def __init__(self):
        self.series_type = 'volume'
        self.source = 'trades'


class Midpoint(Feature):
    def __init__(self):
        self.series_type = 'midpoint'
        self.source = 'orderbook'
        self.param = 1
        self.param_name = 'btc_depth'
        self.lookforward_window = None

    def lookforward(self, window):
        self.lookforward_window = window
        return self

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        prices = super(Midpoint, self).get_data(
            db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        if self.lookforward_window is not None:
            prices = prices.shift(-1 * self.lookforward_window)

        return prices


class MidpointDiff(Midpoint):
    """
    Midpoint price change feature. Without the lookforward call, this represents the
    absolute USD price change in the last hour. With the lookforward call, this
    represents the change in the next period.
    """

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        prices = super(Midpoint, self).get_data(
            db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        price_diffs = prices.diff()

        if self.lookforward_window is not None:
            price_diffs = price_diffs.shift(-1 * self.lookforward_window)

        return price_diffs


class SimpleReturns(Midpoint):
    """
    Is simple return strictly (x + 1) / x? or is it ((x + 2) / x) - 1?
    Let's just go with the -1 version or now because that's what we're really looking
    for.
    """

    def __init__(self):
        super(SimpleReturns, self).__init__()

        self.lookback_window = None
        self.lookforward_window = None

    def lookback(self, window):
        assert self.lookforward_window is None

        self.lookback_window = window
        return self

    def lookforward(self, window):
        assert self.lookback_window is None

        self.lookforward_window = window
        return self

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        prices = super(SimpleReturns, self).get_data(
            db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        simple_returns = (prices / prices.shift(1)) - 1

        # TODO: Support cumulative lookforward/lookbacks.
        # Currently ml.data.processing doesn't have a fast functiong for getting
        # simple return lookforward/backs.
        if self.lookforward_window is not None:
            simple_returns = simple_returns.shift(-1 * self.lookforward_window)

        simple_returns.name = self._get_pretty_name()

        return simple_returns


class LogReturns(Midpoint):
    def __init__(self):
        super(LogReturns, self).__init__()

        self.lookback_window = None
        self.lookforward_window = None

    def lookback(self, window):
        assert self.lookforward_window is None

        self.lookback_window = window
        return self

    def lookforward(self, window):
        assert self.lookback_window is None

        self.lookforward_window = window
        return self

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        prices = super(LogReturns, self).get_data(
            db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        log_prices = np.log(prices)
        log_returns = pd.Series(np.diff(log_prices), index=prices.index[1:])

        if self.lookback_window is not None:
            log_returns = processing.quick_past_returns_series(
                log_returns,
                self.lookback_window,
            )
        elif self.lookforward_window is not None:
            log_returns = processing.future_returns_series(
                log_returns,
                self.lookforward_window,
            )

        log_returns.name = self._get_pretty_name()

        return log_returns

    def _get_pretty_name(self):
        pretty_name = '%s %s %s %s-%s' % (
            type(self).__name__,
            self.exchange_name.capitalize(),
            self.resolution,
            self.param,
            self.param_name,
        )

        if self.lookback_window is not None:
            pretty_name += ' %s-step-lookback' % self.lookback_window
        elif self.lookforward_window is not None:
            pretty_name += ' %s-step-lookforward' % self.lookforward_window

        return pretty_name


class OrderbookStrength(Feature):
    def __init__(self):
        self.source = 'orderbook'
        self.param = 1
        self.param_name = 'usd_slippage'

    def slippage(self, slippage_amount):
        self.param = slippage_amount
        return self


class BidStrength(OrderbookStrength):
    def __init__(self):
        super(BidStrength, self).__init__()
        self.series_type = 'bid_strength'


class AskStrength(OrderbookStrength):
    def __init__(self):
        super(AskStrength, self).__init__()
        self.series_type = 'ask_strength'


class BidStrengthUSD(OrderbookStrength):
    def __init__(self):
        super(BidStrengthUSD, self).__init__()
        self.series_type = 'bid_strength_usd'


class AskStrengthUSD(OrderbookStrength):
    def __init__(self):
        super(AskStrengthUSD, self).__init__()
        self.series_type = 'ask_strength_usd'


class Volatility(LogReturns):
    """
    Volatility of a price series, here defined as standard deviation of log returns in a
    rolling window. We could also try MAD as an alternative.

    There are issues with this subclassing, because this class still has the lookback/
    lookforward methods on it. We might be able to find an intuitive/meaningful way to
    repurpose these functions here, but for now they'll just throw exceptions if they're
    used.
    """
    def lookback(self, lookback_window):
        raise NotImplementedError

    def lookforward(self, lookback_window):
        raise NotImplementedError

    def window(self, lookback_window):
        self._window = lookback_window
        return self

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        log_returns = super(Volatility, self).get_data(
            db,
            start,
            end,
            interpolate,
            max_interpolate,
            cache,
        )

        volatility_series = log_returns.rolling(self._window).std()

        return volatility_series


class Quote(Feature):
    def __init__(self):
        self.source = 'orderbook'
        self.param = 1
        self.param_name = 'btc_depth'

    def slippage(self, slippage_amount):
        self.param = slippage_amount
        return self


class Ask(Quote):
    def __init__(self):
        super(Ask, self).__init__()
        self.series_type = 'ask'


class Bid(Quote):
    def __init__(self):
        super(Bid, self).__init__()
        self.series_type = 'bid'


class Spread(Quote):
    def __init__(self):
        super(Spread, self).__init__()
        self.series_type = 'spread'


class VWAP(Feature):
    def __init__(self):
        self.series_type = 'vwap'
        self.source = 'trades'


class FV(Feature):
    """
    A weighted average of multiple exchange quotes at a depth. Used like this:
        FV().bitstamp().itbit().coinbase().hour().get_data(db, start, end)

    Implementation notes:
    - This requires you to call the time-componenent after all of the exchange
      components, but I think that's ok for now.
    - This feature should give a near-identical value to what the bots would trade
      against iff all exchanges included in this feature are fully funded at that
      moment.
    - One notable difference is the bots tolerate one missing exchange, this doesn't.
      That could potentially be a different feature.
    - I used the opposite approach to InterExchangeSpread, choosing to reimplement all
      the interface methods instead of using clever inheritance to cut down on code.
      Unsure which approach I like more.
    """

    def __init__(self):
        self.series_type = 'ask'
        self.source = 'orderbook'
        self.param = 1
        self.param_name = 'btc_depth'
        self.bid_features = []
        self.ask_features = []

    def slippage(self, slippage_amount):
        self.param = slippage_amount
        self.bid_features = [b.slippage(slippage_amount) for b in self.bid_features]
        self.ask_features = [a.slippage(slippage_amount) for a in self.ask_features]

        return self

    def bitstamp(self):
        self.bid_features.append(Bid().bitstamp())
        self.ask_features.append(Ask().bitstamp())

        return self

    def bitfinex(self):
        self.bid_features.append(Bid().bitfinex())
        self.ask_features.append(Ask().bitfinex())

        return self

    def kraken(self):
        self.bid_features.append(Bid().kraken())
        self.ask_features.append(Ask().kraken())

        return self

    def itbit(self):
        self.bid_features.append(Bid().itbit())
        self.ask_features.append(Ask().itbit())

        return self

    def okcoin(self):
        self.bid_features.append(Bid().okcoin())
        self.ask_features.append(Ask().okcoin())

        return self

    def coinbase(self):
        self.bid_features.append(Bid().coinbase())
        self.ask_features.append(Ask().coinbase())

        return self

    def gemini(self):
        self.bid_features.append(Bid().gemini())
        self.ask_features.append(Ask().gemini())

        return self

    def hour(self):
        self.bid_features = [b.hour() for b in self.bid_features]
        self.ask_features = [a.hour() for a in self.ask_features]
        self.resolution = '1hr'

        return self

    def ten_min(self):
        self.bid_features = [b.ten_min() for b in self.bid_features]
        self.ask_features = [a.ten_min() for a in self.ask_features]
        self.resolution = '10m'

        return self

    def five_min(self):
        self.bid_features = [b.five_min() for b in self.bid_features]
        self.ask_features = [a.five_min() for a in self.ask_features]
        self.resolution = '5m'

        return self

    def get_data(self, db, start, end, interpolate=False, max_interpolate=None, cache=None):
        """
        The logic in gryphon-fury is: get the bid participating exchanges, average their
        bid quotes weighted with their given weights, do the same for asks, and then
        average the two averages.
        """

        bid_data = []
        bid_weight = 0.0

        for b in self.bid_features:
            data = b.get_data(db, start, end, interpolate, max_interpolate, cache)
            weight = self.fv_weights[b.exchange_name.upper()]
            data = data * weight

            bid_data.append(data)
            bid_weight += weight

        bid_data = pd.concat(bid_data, axis=1)
        bid_data = bid_data.dropna()  # Real CFV would allow one missing exchange.
        bid_fv = bid_data.T.sum() / bid_weight

        ask_data = []
        ask_weight = 0.0  # Technically ask/bid weight are the same in this version.

        for a in self.ask_features:
            data = a.get_data(db, start, end, interpolate, max_interpolate, cache)
            weight = self.fv_weights[a.exchange_name.upper()]
            data = data * weight

            ask_data.append(data)
            ask_weight += weight

        ask_data = pd.concat(ask_data, axis=1)
        ask_data = ask_data.dropna()
        ask_fv = ask_data.T.sum() / ask_weight

        fv = (bid_fv + ask_fv) / 2

        return fv


class FV3(FV):
    """
    The fundamental value given with weights determined in Sept 2016, approximately
    each exchange's volume share in the previous six months.
    """
    fv_weights = {
        'COINBASE': 0.25,
        'BITSTAMP': 0.20,
        'ITBIT': 0.25,
        'GEMINI': 0.05,
        'KRAKEN': 0.25,
        'BITFINEX': 0.0,
    }


class FV2(FV):
    """
    The fundamental value using our exchange max balances (divided by the total) as
    weights.
    """
    fv_weights = {
        'BITFINEX': 0.228438,
        'BITSTAMP': 0.27972,
        'CAVIRTEX': 0.11655,
        'COINBASE': 0.228438,
        'COINBASE_CAD': 0.004662,
        'COINSETTER': 0.004662,
        'ITBIT': 0.051282,
        'KRAKEN': 0.04662,
        'OKCOIN': 0.034965,
        'QUADRIGA': 0.004662,
    }


class ArbFeature(Feature):
    def __init__(self):
        self.source = 'orderbook'
        self.param = None  # Unused.
        self.param_name = None  # Unused.

        self.primary_exchange = None
        self.secondary_exchange = None

    def bitstamp(self):
        return self._exchange_function_call('bitstamp')

    def bitfinex(self):
        return self._exchange_function_call('bitfinex')

    def kraken(self):
        return self._exchange_function_call('kraken')

    def itbit(self):
        return self._exchange_function_call('itbit')

    def okcoin(self):
        return self._exchange_function_call('okcoin')

    def coinbase(self):
        return self._exchange_function_call('coinbase')

    def gemini(self):
        return self._exchange_function_call('gemini')

    def _exchange_function_call(self, exchange_name):
        if self.primary_exchange is None:
            self.primary_exchange = exchange_name
        elif self.secondary_exchange is None:
            self.secondary_exchange = exchange_name
        else:
            raise Exception('Called an exchange function three times!')

        return self

    def _get_pretty_name(self):
        return '%s %s %s %s' % (
            type(self).__name__,
            self.primary_exchange.capitalize(),
            self.secondary_exchange.capitalize(),
            self.resolution,
        )

    def _get_series_name(self):
        series_name = '%s-%s-%s-%s-%s' % (
            self.series_type,
            self.primary_exchange,
            self.secondary_exchange,
            self.source,
            self.resolution,
        )

        return series_name


class SignedArbRevenue(ArbFeature):
    def __init__(self):
        super(SignedArbRevenue, self).__init__()
        self.series_type = 'signed_arb_revenue_usd'


class SignedArbVolumeUSD(ArbFeature):
    def __init__(self):
        super(SignedArbVolumeUSD, self).__init__()
        self.series_type = 'signed_arb_volume_usd'


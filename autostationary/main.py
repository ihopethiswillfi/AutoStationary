import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from scipy.stats import boxcox
from scipy.special import inv_boxcox

import warnings

warnings.simplefilter(action='ignore', category=InterpolationWarning)


class AutoStationary:
    """
    Toolkit to easily transform and inverse_transform a series to a stationary version and back.
    Uses transforming and differencing.  Automatically determines sensible parameters.
    """

    def __init__(self, data, do_nothing=False, warnings_enabled=True):
        """input: 1D array or a Pandas Series
           do_nothing:  Can be used to temporarily disable all data manipulation.  This can be useful when you
                        want to check the behavior of your code on the original, non-stationarized data.
        """

        if not do_nothing:
            if not isinstance(data, pd.Series):
                # create series with a dummy datetimeindex
                data = pd.Series(data, index=pd.DatetimeIndex(start=0, freq='1d', periods=len(data)))
                self._input_was_array = True
            else:
                self._input_was_array = False

        self._do_nothing = do_nothing
        self._data = data
        self._data_index = data.index
        self._tdata = None
        self._diff_order = None
        self._boxcox_lambda = None
        self._boxcox_shift = 0
        self._stationary = None
        self._transformed = False
        self._adf_result = None
        self._kpss_result = None
        self.warnings_enabled = warnings_enabled

    def _is_stationary_ADF(self, data, critical_value='5%'):
        assert (critical_value in ['1%', '5%', '10%'])
        result = adfuller(data, autolag='AIC')
        teststat = result[0]
        criticv = result[4][critical_value]
        self._adf_result = result
        if teststat < criticv:
            return True
        else:
            return False

    def _is_stationary_KPSS(self, data, critical_value='5%'):
        assert (critical_value in ['1%', '5%', '10%'])
        result = kpss(data, regression='c')
        teststat = result[0]
        criticv = result[3][critical_value]
        self._kpss_result = result
        if teststat < criticv:
            return True
        else:
            return False

    def _is_stationary(self, data, critical_value='5%'):
        """Returns whether the data is stationary according to ADF and KPSS.  If one of the two tests fail, returns False"""
        assert (critical_value in ['1%', '5%', '10%'])
        res_adf = self._is_stationary_ADF(data, critical_value)
        res_kpss = self._is_stationary_KPSS(data, critical_value)
        if res_adf and res_kpss:
            self._stationary = True
            return True
        else:
            self._stationary = False
            return False

    def transform(self, boxcox_transform=True, diff_orders='auto', enforce_stationarity=True, critical_value='5%'):
        """
        If the data is not stationary, returns a stationary version of the data.
        Used techniques are Boxcox transform and differencing.
        Both ADF and KPSS tests have to be passed to consider the data stationary.

        boxcox_transformation       Deal with unstable variance by transforming the data before applying differencing.
                                    True, False, or float.
                                    Set to False to disable.
                                    Set to 0.0 to do a log transform.
                                    Set to True to automatically determine the optimal value.

        diff_orders                 int:  difference the series by the specified order
                                    list of ints: try all orders specified in the list
                                    'auto': (default)  uses np.arange(1, 25)

        enforce_stationarity        If the data can't be transformed to stationary, throws an error if set to True.
                                    If set to False, will return the data with diff_orders[0] differencing.
                                    Note that if this setting is disabled and you have disabled warning messages as well,
                                    the function may return non-stationary data without alerting you.

        critical_value              Required confidence for the ADF and KPSS stationarity tests.
                                    String.  '1%', 5%' (default) or '10%'.
        """

        # input validation
        if diff_orders == 'auto':
            orders = np.arange(1, 25)
        elif isinstance(diff_orders, int):
            orders = [diff_orders]
        elif isinstance(diff_orders, list):
            assert (isinstance(diff_orders[0], int))
        if self._transformed:
            raise ValueError('Data already transformed!', UserWarning)

        # do nothing
        if self._do_nothing:
            return self._data

        # if already stationary, do nothing
        if self._is_stationary(self._data):
            if self.warnings_enabled:
                warnings.warn('Data already stationary.  Returning original data.', UserWarning)
            self._transformed = False
            return self._data

        # boxcox transformation
        tdata = self._data
        if boxcox_transform:
            # boxcox requires only positive data
            lowest = np.min(tdata)
            if lowest <= 0:
                self._boxcox_shift = np.abs(lowest) + 1
                tdata = tdata + self._boxcox_shift

            if isinstance(boxcox_transform, float):
                # lambda was specified
                self._boxcox_lambda = boxcox_transform
                tdata = boxcox(tdata, boxcox_transform)
            else:
                # find best lambda
                tdata, lmbda = boxcox(tdata)
                self._boxcox_lambda = lmbda
            tdata = pd.Series(tdata, index=self._data_index)  # store transformed data
            self._tdata = tdata

        # find best order
        for o in orders:
            datadiff = np.roll(tdata, o)
            datadiff = tdata - datadiff
            datadiff = datadiff[o:]
            if self._is_stationary(datadiff, critical_value):
                self._diff_order = o
                self._transformed = True
                if self._input_was_array:
                    return datadiff.values
                else:
                    return datadiff

        # if failed to make it stationary
        self.stationary = False
        if enforce_stationarity:
            raise ValueError('Could not make the data stationary with given parameters.')
        else:
            if self.warnings_enabled:
                warnings.warn(f'Could not make the data stationary with given parameters.  Returning difference with order {orders[0]}.', UserWarning)
            self._diff_order = orders[0]
            datadiff = np.roll(tdata, self._diff_order)
            datadiff = tdata - datadiff
            datadiff = datadiff[self._diff_order:]
            self._transformed = True
            if self._input_was_array:
                return datadiff.values
            else:
                return datadiff

    def inverse_transform(self, diffdata):
        if self._do_nothing:
            return diffdata

        if isinstance(diffdata, pd.Series):
            diffdata = diffdata.values

        if not self._transformed:
            if self.warnings_enabled:
                warnings.warn('The original data was never transformed.  Returning the same data you just passed.', UserWarning)
            return diffdata

        # insert the start of the original series
        if self._tdata is None:
            diffdata = np.insert(diffdata, [0], self._data[:self._diff_order])
        else:
            # a boxcox was applied
            diffdata = np.insert(diffdata, [0], self._tdata[:self._diff_order])

        # append 0's as padding if required
        if (len(diffdata) % self._diff_order) == 0:
            padlen = 0
        else:
            padlen = self._diff_order - (len(diffdata) % self._diff_order)
        diffdata = np.append(diffdata, np.zeros(padlen))

        # cumsum while accounting for order
        nrows = len(diffdata) // self._diff_order
        diffdata = np.reshape(diffdata, (nrows, self._diff_order))
        inv = np.cumsum(diffdata, axis=0)

        # cleanup
        inv = inv.flatten()
        if padlen > 0:
            inv = inv[:-padlen]

        # inverse boxcox
        if self._boxcox_lambda is not None:
            inv = inv_boxcox(inv, self._boxcox_lambda)

        # inverse boxcox shift
        if self._boxcox_shift != 0:
            inv = inv - self._boxcox_shift

        # add index if it was a series originally
        if not self._input_was_array:
            # we must auto-extend the index if the data is now longer.
            idxorig = self._data_index
            idxtoadd = [idxorig.index[-1] + i + 1 for i in range(len(inv) - len(idxorig))]
            idxnew = idxorig.union(idxtoadd)
            # but we must also handle a potentially shorter array
            idxnew = idxnew[:len(inv)]
            inv = pd.Series(inv, index=idxnew)
        return inv

    def summary(self):
        """returns a summary after calling transform()"""
        if not self._do_nothing:
            return ({
                'transformed': self._transformed,
                'stationary': self._stationary,
                'diff_order': self._diff_order,
                'boxcox_lambda': self._boxcox_lambda,
                'boxcox_shift': self._boxcox_shift,
                'adf_result': pd.Series(self._adf_result[:5], index=['test_statistic', 'p', 'n_lags', 'n_obs', 'critical_value']).to_dict(),
                'kpss_result': pd.Series(self._kpss_result[:4], index=['test statistic', 'p', 'n_lags', 'critical_value']).to_dict()
            })
        else:
            return {'warning': 'data was not transformed in any way cause user set do_nothing.'}

# data = pd.read_csv('Datasets/AirPassengers.csv')
# print(data['#Passengers'].values)
# ass = AutoStationary(data['#Passengers'].values)
# print(ass._is_stationary_ADF(data['#Passengers'].values))
# print(ass._is_stationary_KPSS(data['#Passengers'].values))
# diff = ass.transform(boxcox_transform=True, critical_value='5%')
# print(diff)
# print(ass.summary())
# #print(ass.difference(critical_value='1%'))
# print(ass._is_stationary_ADF(diff))
# print(ass._is_stationary_KPSS(diff))
# print(diff)
# print(ass.inverse_transform(diff))


# file = pd.read_csv('Datasets/sinewave.csv')['sinewave'].values
# print(file[:5])
# astat = AutoStationary(file)
# tfile = astat.transform()
# print(astat.summary())
# print(tfile[:5])
# rfile = astat.inverse_transform(tfile)
# print(rfile[:5])


# data = pd.read_csv('Datasets/AirPassengers.csv')
# data['Month'] = pd.to_datetime(data['Month'])
# data = data.set_index('Month')
# data = data['#Passengers']  # make the df a series
# print(data.head())
# print(data.tail())
# print('-'*10)
# ass = AutoStationary(data)
# diff = ass.transform(boxcox_transform=True, critical_value='5%')
# print(ass.summary())
# print(diff.head())
# print(diff.tail())
# print('-'*10)
# inv = ass.inverse_transform(diff)
# print(inv.head())
# print(inv.tail())
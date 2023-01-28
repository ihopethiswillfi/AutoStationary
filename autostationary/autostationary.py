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

    def __init__(self, data, enabled=True, warnings_enabled=True, critical_value='5%'):
        """input: 1D array or a Pandas Series
           enabled:  If False, temporarily disable all data manipulation.  This can be useful when you
                     want to check the behavior of your code on the original, non-stationarized data.
           critical_value              Required confidence for the ADF and KPSS stationarity tests.
                                       String.  '1%', 5%' (default) or '10%'.
        """
        if critical_value not in ['1%', '5%', '10%']:
            raise ValueError(f"critical value must be in ['1%', '5%', '10%']")

        if enabled:
            if not isinstance(data, pd.Series):
                # create series with a dummy datetimeindex
                data = pd.Series(data, index=pd.date_range(start=0, freq='1d', periods=len(data)))
                self._input_was_array = True
            else:
                self._input_was_array = False

        self._enabled = enabled
        self._data = data
        self._data_index = data.index
        self._tdata = None
        self._diff_order = None
        self._boxcox_lambda = None
        self._boxcox_shift = 0
        self._transformed = False
        self.stationary = None
        self.critical_value = critical_value
        self.warnings_enabled = warnings_enabled

    def is_stationary_ADF(self, data, return_result=False):
        result = adfuller(data, autolag='AIC')
        teststat = result[0]
        criticv = result[4][self.critical_value]
        if teststat < criticv:
            if not return_result:
                return True
            else:
                return True, result
        else:
            if not return_result:
                return False
            else:
                return False, result

    def is_stationary_KPSS(self, data, return_result=False):
        result = kpss(data, regression='c')
        teststat = result[0]
        criticv = result[3][self.critical_value]
        if teststat < criticv:
            if not return_result:
                return True
            else:
                return True, result
        else:
            if not return_result:
                return False
            else:
                return False, result

    def is_stationary(self, data):
        """Returns whether the data is stationary according to ADF and KPSS.  If one of the two tests fail, returns False"""
        res_adf = self.is_stationary_ADF(data)
        res_kpss = self.is_stationary_KPSS(data)
        if res_adf and res_kpss:
            self.stationary = True
            return True
        else:
            self.stationary = False
            return False

    def transform(self, boxcox_transform=True, diff_orders='auto', enforce_stationarity=True):
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
        if not self._enabled:
            return self._data

        # if already stationary, do nothing
        if self.is_stationary(self._data):
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
            if self.is_stationary(datadiff):
                self._diff_order = o
                self._transformed = True
                if self._input_was_array:
                    return datadiff.values
                else:
                    return datadiff

        # failed to make it stationary
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
        if not self._enabled:
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

        if not self._enabled:
            return {'warning': 'data was not transformed in any way cause user set enabled.'}
        else:
            _, adf_result = self.is_stationary_ADF(self._data, return_result=True)
            _, kpss_result = self.is_stationary_KPSS(self._data, return_result=True)

            return ({'transformed': self._transformed,
                    'stationary': self.stationary,
                    'diff_order': self._diff_order,
                    'boxcox_lambda': self._boxcox_lambda,
                    'boxcox_shift': self._boxcox_shift,
                    'adf_result': pd.Series(adf_result[:5], index=['test_statistic', 'p', 'n_lags', 'n_obs', 'critical_value']).to_dict(),
                    'kpss_result': pd.Series(kpss_result[:4], index=['test statistic', 'p', 'n_lags', 'critical_value']).to_dict()
                    })



if __name__ == '__main__':
    data = pd.read_csv('../datasets/airpassengers.csv')
    col = '#Passengers'

    print(data[col].values)
    ast = AutoStationary(data[col].values)
    print(ast.is_stationary_ADF(data[col].values))
    print(ast.is_stationary_KPSS(data[col].values))

    diff = ast.transform(boxcox_transform=True)
    print(diff)
    print(ast.summary())
    #print(ast.difference(critical_value='1%'))
    print(ast.is_stationary_ADF(diff))
    print(ast.is_stationary_KPSS(diff))

    print(ast.inverse_transform(diff))


    # file = pd.read_csv('datasets/sinewave.csv')['sinewave'].values
    # print(file[:5])
    # astat = AutoStationary(file)
    # tfile = astat.transform()
    # print(astat.summary())
    # print(tfile[:5])
    # rfile = astat.inverse_transform(tfile)
    # print(rfile[:5])


    # data = pd.read_csv('datasets/airPassengers.csv')
    # data['Month'] = pd.to_datetime(data['Month'])
    # data = data.set_index('Month')
    # data = data['#Passengers']  # make the df a series
    # print(data.head())
    # print(data.tail())
    # print('-'*10)
    # ast = AutoStationary(data)
    # diff = ast.transform(boxcox_transform=True)
    # print(ast.summary())
    # print(diff.head())
    # print(diff.tail())
    # print('-'*10)
    # inv = ast.inverse_transform(diff)
    # print(inv.head())
    # print(inv.tail())
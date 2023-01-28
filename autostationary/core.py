import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from scipy.stats import boxcox
from scipy.special import inv_boxcox

import warnings

warnings.simplefilter(action='ignore', category=InterpolationWarning)


class AutoStationary:
    """Transform and inverse_transform a time-series to a stationary version and back.

    Uses boxcox transform and differencing.  Sensible parameters are automatically determined and applied
    with respect to the provided input parameters.

    The inverse_transform() is especially practical for cases in which you have a forecasting model which
    takes transformed data as input.  The output of the model in this case will also be in a transformed
    state, so you need to first inverse_transform() it before you can make sense of your forecast.
    """

    def __init__(self, arr, enabled=True, verbose=True, critical_value='5%'):
        """
        :param arr:             1-dimensional numpy array or pandas Series
        :param enabled:         bool.  If False, temporarily disables all data manipulation.  This can be
                                useful when you want to check the behavior of your code on the original,
                                unmodified data.
        :param verbose:         bool.
        :param critical_value:  str.  Required confidence for the ADF and KPSS stationarity tests.
                                Allowed values: ['1%', 5%', '10%'].  Default: '5%'
        """

        if critical_value not in ['1%', '5%', '10%']:
            raise ValueError(f"critical value must be in ['1%', '5%', '10%']")

        if enabled:
            if not isinstance(arr, pd.Series):
                # create series with a dummy datetimeindex
                arr = pd.Series(arr, index=pd.date_range(start=0, freq='1d', periods=len(arr)))
                self._input_was_array = True
            else:
                self._input_was_array = False

        self._enabled = enabled
        self._arr = arr
        self._arr_index = arr.index
        self._tarr = None
        self._diff_order = None
        self._boxcox_lambda = None
        self._boxcox_shift = 0
        self._transformed = False
        self.stationary = None
        self.critical_value = critical_value
        self.warnings_enabled = verbose

    def is_stationary_ADF(self, arr, return_result=False):
        """Returns whether the array is stationary according to the ADF test.

        ADF = Augmented Dickey Fuller Test.

        :param arr:             1-dimensional numpy array or pandas Series
        :param return_result:   bool: if False, returns a bool.  if True, returns a (bool, dict) tuple.
        :return:                bool or tuple, depending on the `return_result` parameter.
        """
        result = adfuller(arr, autolag='AIC')
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

    def is_stationary_KPSS(self, arr, return_result=False):
        """Returns whether the array is stationary according to the KPSS test.

        KPSS = Kwiatkowski–Phillips–Schmidt–Shin Test.

        :param arr:             1-dimensional numpy array or pandas Series
        :param return_result:   bool: if False, returns a bool.  if True, returns a (bool, dict) tuple.
        :return:                bool or tuple, depending on the `return_result` parameter.
        """
        result = kpss(arr, regression='c')
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

    def is_stationary(self, arr):
        """Returns whether the array is stationary according to both ADF and KPSS.

        :param arr:             1-dimensional numpy array or pandas Series
        :return:                bool
        """

        res_adf = self.is_stationary_ADF(arr)
        res_kpss = self.is_stationary_KPSS(arr)
        if res_adf and res_kpss:
            self.stationary = True
            return True
        else:
            self.stationary = False
            return False

    def transform(self, boxcox_transform=True, diff_orders='auto', enforce_stationarity=True):
        """Returns a stationary version of the array.

        Used techniques are Boxcox transform and differencing of the n-th order.
        Both ADF and KPSS stationary tests have to pass before we consider the array stationary.

        :param boxcox_transform:        Deal with unstable variance by transforming the array before
                                        applying differencing.  True, False, or float.
                                            Set to False to disable boxcox transformation.
                                            Set to 0.0 to do a log transform.
                                            Set to True to automatically determine the optimal (float) value.

        :param diff_orders:             Determines the maximum order of differencing.
                                            int:                difference the series by the specified order
                                            list of ints:       try all orders specified in the list
                                            'auto' (default):   uses np.arange(1, 25)

        :param enforce_stationarity:    If True, raise an Exception if we fail to make array stationary.
                                        If False, instead of an Exception, the array will return with
                                        diff_orders[0] differencing.
                                        Note: if False, and verbose is also False, you may be returned
                                        a non-stationary array as output without being aware of it!

        :return:                        Stationary version of the array (For edge cases read above).
        """

        # input validation
        if diff_orders == 'auto':
            orders = np.arange(1, 25)
        elif isinstance(diff_orders, int):
            orders = [diff_orders]
        elif isinstance(diff_orders, list):
            if not (isinstance(diff_orders[0], int)):
                raise ValueError('diff_orders must be integers.')
        if self._transformed:
            raise ValueError('array already transformed!', UserWarning)

        # do nothing
        if not self._enabled:
            return self._arr

        # if already stationary, do nothing
        if self.is_stationary(self._arr):
            if self.warnings_enabled:
                warnings.warn('array already stationary.  Returning original array.', UserWarning)
            self._transformed = False
            return self._arr

        # boxcox transformation
        tarr = self._arr
        if boxcox_transform:
            # boxcox requires a strictly positive array
            lowest = np.min(tarr)
            if lowest <= 0:
                self._boxcox_shift = np.abs(lowest) + 1
                tarr = tarr + self._boxcox_shift

            if isinstance(boxcox_transform, float):
                # lambda was specified
                self._boxcox_lambda = boxcox_transform
                tarr = boxcox(tarr, boxcox_transform)
            else:
                # find best lambda
                tarr, lmbda = boxcox(tarr)
                self._boxcox_lambda = lmbda
            tarr = pd.Series(tarr, index=self._arr_index)  # store transformed arr
            self._tarr = tarr

        # find best order
        for o in orders:
            arrdiff = np.roll(tarr, o)
            arrdiff = tarr - arrdiff
            arrdiff = arrdiff[o:]
            if self.is_stationary(arrdiff):
                self._diff_order = o
                self._transformed = True
                if self._input_was_array:
                    return arrdiff.values
                else:
                    return arrdiff

        # failed to make it stationary
        self.stationary = False
        if enforce_stationarity:
            raise ValueError('Could not make the arr stationary with given parameters.')
        else:
            if self.warnings_enabled:
                warnings.warn(f'Could not make the arr stationary with given parameters.  Returning difference with order {orders[0]}.', UserWarning)
            self._diff_order = orders[0]
            arrdiff = np.roll(tarr, self._diff_order)
            arrdiff = tarr - arrdiff
            arrdiff = arrdiff[self._diff_order:]
            self._transformed = True
            if self._input_was_array:
                return arrdiff.values
            else:
                return arrdiff

    def inverse_transform(self, diffarr):
        if not self._enabled:
            return diffarr

        if isinstance(diffarr, pd.Series):
            diffarr = diffarr.values

        if not self._transformed:
            if self.warnings_enabled:
                warnings.warn('The original array was never transformed.  Returning the same array you just passed.', UserWarning)
            return diffarr

        # insert the start of the original series
        if self._tarr is None:
            diffarr = np.insert(diffarr, [0], self._arr[:self._diff_order])
        else:
            # a boxcox was applied
            diffarr = np.insert(diffarr, [0], self._tarr[:self._diff_order])

        # append 0's as padding if required
        if (len(diffarr) % self._diff_order) == 0:
            padlen = 0
        else:
            padlen = self._diff_order - (len(diffarr) % self._diff_order)
        diffarr = np.append(diffarr, np.zeros(padlen))

        # cumsum while accounting for order
        nrows = len(diffarr) // self._diff_order
        diffarr = np.reshape(diffarr, (nrows, self._diff_order))
        inv = np.cumsum(diffarr, axis=0)

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
            # we must auto-extend the index if the arr is now longer.
            idxorig = self._arr_index
            idxtoadd = [idxorig.index[-1] + i + 1 for i in range(len(inv) - len(idxorig))]
            idxnew = idxorig.union(idxtoadd)
            # but we must also handle a potentially shorter array
            idxnew = idxnew[:len(inv)]
            inv = pd.Series(inv, index=idxnew)
        return inv

    def summary(self):
        """returns a summary of the current state of the array"""

        if not self._enabled:
            return {'warning': 'arr was not transformed in any way cause user set enabled.'}
        else:
            _, adf_result = self.is_stationary_ADF(self._arr, return_result=True)
            _, kpss_result = self.is_stationary_KPSS(self._arr, return_result=True)

            return ({'transformed': self._transformed,
                    'stationary': self.stationary,
                    'diff_order': self._diff_order,
                    'boxcox_lambda': self._boxcox_lambda,
                    'boxcox_shift': self._boxcox_shift,
                    'adf_result': pd.Series(adf_result[:5], index=['test_statistic', 'p', 'n_lags', 'n_obs', 'critical_value']).to_dict(),
                    'kpss_result': pd.Series(kpss_result[:4], index=['test statistic', 'p', 'n_lags', 'critical_value']).to_dict()
                    })

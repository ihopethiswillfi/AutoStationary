import numpy as np
import pandas as pd
from context import autostationary

if __name__ == '__main__':
    # load series
    data = pd.read_csv('../datasets/airpassengers.csv').set_index('Month')
    col = '#Passengers'
    srs = data[col]

    # instantiate
    ast = autostationary.AutoStationary(srs)

    # original series must not be stationary
    assert not ast.is_stationary_ADF(srs)
    assert not ast.is_stationary_KPSS(srs)
    assert not ast.is_stationary(srs)
    assert not ast.summary()['transformed']
    assert not ast.summary()['stationary']

    # transformed series must be stationary
    trsrs = ast.transform()
    assert ast.is_stationary_ADF(trsrs)
    assert ast.is_stationary_KPSS(trsrs)
    assert ast.is_stationary(trsrs)
    assert ast.summary()['transformed']
    assert ast.summary()['stationary']
    assert ast.summary()['diff_order'] == 2

    # inverse transform of trsrs must equal srs again
    assert np.alltrue(np.isclose(ast.inverse_transform(trsrs), srs))

    print('Test completed successfully.')

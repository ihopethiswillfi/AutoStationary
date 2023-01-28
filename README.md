# AutoStationary
Toolkit to easily transform and inverse_transform a **time-series** to a stationary version and back. 

- Uses boxcox transform and differencing.
- Sensible parameters are automatically determined and applied with respect to the provided input parameters.

The `inverse_transform()` is especially practical for cases in which you have a forecasting model which
takes transformed data as input.  The output of the model in this case will also be
in a transformed state, so you need to first inverse_transform() it before you can make sense of
your forecast.

I originally created AutoStationary in early 2019 when I was working on several time-series forecasting projects,
and I wanted to speed up prototyping for new datasets.

The code is certainly NOT ready for production.  I consider it still a work-in-progress.

## Usage
```python
from autostationary import AutoStationary

# transform
arr = [pandas series or numpy array]
ast = AutoStationary(arr, critical_value='5%')
transformed_arr = ast.transform()

# inverse transform, useful when your model trained and predicted on transformed data.
transformed_preds = model_trained_on_transformed_labels.predict(X)
preds = ast.inverse_transform(transformed_preds)

# additional functions
ast.is_stationary_ADF(arr)  # bool: is arr stationary according to ADF?
ast.is_stationary_KPSS(arr) # bool: is arr stationary according to KPSS?
ast.summary()  # returns current state of the array in the class
```
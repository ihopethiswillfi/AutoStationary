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
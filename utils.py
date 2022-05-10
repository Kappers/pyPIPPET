'''
'''
from __future__ import annotations
import numpy as np

def template(model: PIPPET) -> np.ndarray:
    ''' Collapse expectation templates into a numpy array '''
    from scipy.stats import norm
    from PIPPET import WIPPET

    if isinstance(model, WIPPET):
        # Wrapped expectations
        ts = np.arange(-np.pi, np.pi, model.params.dt)
        temp = np.zeros((model.n_streams, ts.size))
    else:
        # Expectations along a line
        ts = model.ts
        temp = np.zeros((model.n_streams, model.n_ts))

    for s_i in range(model.n_streams):
        stream = model.streams[s_i].params
        for i in range(stream.e_means.size):
            pdf = norm.pdf(ts, loc=stream.e_means[i], scale=(stream.e_vars[i])**0.5)
            temp[s_i] += stream.e_lambdas[i] * pdf

    return ts, temp + model.params.lambda_0

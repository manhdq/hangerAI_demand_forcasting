import numpy as np
from sklearn.metrics import mean_absolute_error


def calc_error_metrics(gt, forecasts):
    ##TODO: Add `tracking signal` metric
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * (np.sum(np.abs(gt - forecasts)) / np.sum(gt))

    return round(mae, 3), round(wape, 3)
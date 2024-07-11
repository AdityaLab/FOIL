import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)
import numpy as np

def top_x_percent_avg(values, xx):
    n = len(values)
    top_x_percent = int(xx * n)
    sorted_values = np.sort(values.flatten())[::-1]
    return np.mean(sorted_values[:top_x_percent])

def MAE_W(pred, true, xx):
    # Calculate the mean absolute error for each sample
    abs_diff_per_sample = np.mean(np.abs(pred - true), axis=1)
    return top_x_percent_avg(abs_diff_per_sample, xx)

def MSE_W(pred, true, xx):
    # Calculate the mean squared error for each sample
    squared_diff_per_sample = np.mean((pred - true) ** 2, axis=1)
    return top_x_percent_avg(squared_diff_per_sample, xx)






def MSE_var(pred, true):
    return np.var((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true,xx):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mse_var = MSE_var(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mse_w= MSE_W(pred, true,xx)
    mae_w= MAE_W(pred, true,xx)

    
    return mae,mse,rmse,mape,mspe,mse_var,mse_w,mae_w
from __future__ import division
import numpy as np

# ----------- Gaussian Smoothing (self-defined) ------------------------------------------------------------
''' https://matthew-brett.github.io/teaching/smoothing_intro.html '''

def sigma2fwhm(sigmaa):
	return sigmaa * np.sqrt(8 * np.log(2))


def fwhm2sigma(fwhm):
	return fwhm / np.sqrt(8 * np.log(2))
	

def gauss_smooth_func(x_vals, y_vals, sigmaa):
    smoothed_vals = np.zeros(y_vals.shape)
    
    for i in range(len(y_vals)):
        x_position = x_vals[i]
        kernel = (1.0/np.sqrt(2*np.pi *sigmaa**2)) * np.exp( -(x_vals-x_position)** 2 / (2 * sigmaa ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[i] = np.sum(y_vals * kernel)
        
    return smoothed_vals   

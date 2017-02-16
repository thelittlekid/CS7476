# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:00:06 2017

@author: fantasie
"""

# As usual, a bit of setup

import time
import numpy as np
import matplotlib.pyplot as plt
from code.classifiers.fc_net import *
from code.data_utils import get_CIFAR10_data
from code.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from code.solver import Solver

#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
  

## Load the (preprocessed) CIFAR10 data.
#
#data = get_CIFAR10_data()
#for k, v in data.iteritems():
#  print '%s: ' % k, v.shape
#  
#x = np.random.randn(500, 500) + 10

for p in [0.3, 0.6, 0.75]:
  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

  print 'Running tests with p = ', p
  print 'Mean of input: ', x.mean()
  print 'Mean of train-time output: ', out.mean()
  print 'Mean of test-time output: ', out_test.mean()
  print 'Fraction of train-time output set to zero: ', (out == 0).mean()
  print 'Fraction of test-time output set to zero: ', (out_test == 0).mean()
  print
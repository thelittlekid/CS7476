# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:13:36 2017

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
#
## for auto-reloading external modules
## see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#
#def rel_error(x, y):
#  """ returns relative error """
#  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#  
#
## Load the (preprocessed) CIFAR10 data.
#
#data = get_CIFAR10_data()
#for k, v in data.iteritems():
#  print '%s: ' % k, v.shape
#  
#  
## Check the training-time forward pass by checking means and variances
## of features both before and after batch normalization
#
## Simulate the forward pass for a two-layer network
#N, D1, D2, D3 = 200, 50, 60, 3
#X = np.random.randn(N, D1)
#W1 = np.random.randn(D1, D2)
#W2 = np.random.randn(D2, D3)
#a = np.maximum(0, X.dot(W1)).dot(W2)
#
#print 'Before batch normalization:'
#print '  means: ', a.mean(axis=0)
#print '  stds: ', a.std(axis=0)
#
## Means should be close to zero and stds close to one
#print 'After batch normalization (gamma=1, beta=0)'
#a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
#print '  mean: ', a_norm.mean(axis=0)
#print '  std: ', a_norm.std(axis=0)
#
## Now means should be close to beta and stds close to gamma
#gamma = np.asarray([1.0, 2.0, 3.0])
#beta = np.asarray([11.0, 12.0, 13.0])
#a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
#print 'After batch normalization (nontrivial gamma, beta)'
#print '  means: ', a_norm.mean(axis=0)
#print '  stds: ', a_norm.std(axis=0)
#
#
## Check the test-time forward pass by running the training-time
## forward pass many times to warm up the running averages, and then
## checking the means and variances of activations after a test-time
## forward pass.
#
#N, D1, D2, D3 = 200, 50, 60, 3
#W1 = np.random.randn(D1, D2)
#W2 = np.random.randn(D2, D3)
#
#bn_param = {'mode': 'train'}
#gamma = np.ones(D3)
#beta = np.zeros(D3)
#for t in xrange(50):
#  X = np.random.randn(N, D1)
#  a = np.maximum(0, X.dot(W1)).dot(W2)
#  batchnorm_forward(a, gamma, beta, bn_param)
#bn_param['mode'] = 'test'
#X = np.random.randn(N, D1)
#a = np.maximum(0, X.dot(W1)).dot(W2)
#a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
#
## Means should be close to zero and stds close to one, but will be
## noisier than training-time forward passes.
#print 'After batch normalization (test-time):'
#print '  means: ', a_norm.mean(axis=0)
#print '  stds: ', a_norm.std(axis=0)
#
## Gradient check batchnorm backward pass
#
#N, D = 4, 5
#x = 5 * np.random.randn(N, D) + 12
#gamma = np.random.randn(D)
#beta = np.random.randn(D)
#dout = np.random.randn(N, D)
#
#bn_param = {'mode': 'train'}
#fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
#fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
#fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
#
#dx_num = eval_numerical_gradient_array(fx, x, dout)
#da_num = eval_numerical_gradient_array(fg, gamma, dout)
#db_num = eval_numerical_gradient_array(fb, beta, dout)
#
#_, cache = batchnorm_forward(x, gamma, beta, bn_param)
#dx, dgamma, dbeta = batchnorm_backward(dout, cache)
#print 'dx error: ', rel_error(dx_num, dx)
#print 'dgamma error: ', rel_error(da_num, dgamma)
#print 'dbeta error: ', rel_error(db_num, dbeta)


N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print 'Running check with reg = ', reg
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64,
                            use_batchnorm=True)

  loss, grads = model.loss(X, y)
  print 'Initial loss: ', loss

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
  if reg == 0: print

## Try training a very deep net with batchnorm
#hidden_dims = [100, 100, 100, 100, 100]
#
#num_train = 1000
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#weight_scale = 2e-2
#bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
#model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
#
#bn_solver = Solver(bn_model, small_data,
#                num_epochs=10, batch_size=50,
#                update_rule='sgd',
#                optim_config={
#                  'learning_rate': 1e-3,
#                },
#                verbose=True, print_every=200)
#bn_solver.train()
#
#solver = Solver(model, small_data,
#                num_epochs=10, batch_size=50,
#                update_rule='sgd',
#                optim_config={
#                  'learning_rate': 1e-3,
#                },
#                verbose=True, print_every=200)
#solver.train()
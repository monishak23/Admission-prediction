#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


class logisticReg:
  
  def __init__(self, lr = 0.0001,n_iters =1000):
        self.lr = lr
        self.n_iters = n_iters
        self.slope = None
        self.intp = None

  def fit(self,X,y):
      n_samples, n_features = X.shape
      self.slope = np.zeros(n_features)
      self.intp = 0

      #gd
      for _ in range(self.n_iters):
         
          linear_model = np.dot(X,self.slope) + self.intp
          y_predicted = self.sigmoid(linear_model)

          dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
          db = (1/n_samples) * np.sum(y_predicted - y)

          self.slope -= self.lr * dw
          self.intp -= self.lr * db 

  def predict(self,X):
          linear_model = np.dot(X,self.slope) + self.intp
          y_predicted = self.sigmoid(linear_model)

          y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
          return y_predicted_cls

  def sigmoid(self,x):
    return 1/(1+np.exp(-x))


# In[ ]:





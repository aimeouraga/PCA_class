#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    
    def fit(self, X):
        '''self.fit()
        fits data to PCA class initialized.'''
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

    
    def mean(self):
        '''self.mean()
        returns mean matrix of fitted data.'''
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.' 
        mean = np.sum(self.X, axis=0, keepdims=True)/self.n_samples
        return mean

    
    def std(self):
        '''self.std()
        returns standard deviation matrix of fitted data.'''
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.' 
        std = np.sqrt(np.sum((self.X - self.mean()) ** 2, keepdims=True, axis=0)/(self.n_samples - 1))
        return std

    
    def standardize_data(self):
        '''self.standardize_data()
        returns standardized fitted data.'''
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        standardized_data = (self.X - self.mean())/self.std()
        return standardized_data

    
    def covariance(self):
        '''self.covariance()
        returns covariance matrix of fitted data.'''
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        x = self.standardize_data()
        cov =  (x.T @ x)/(self.n_samples - 1)
        return cov

    
    def eig_vals_vecs(self):
        eig_vals, eig_vecs = eig(self.covariance())

        idx = eig_vals.argsort()[::-1]   
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        return eig_vals, eig_vecs


    def explained_variance(self):
        eig_vals, _ = self.eig_vals_vecs()
        proportions = [100 * eig_val/sum(eig_vals) for eig_val in eig_vals]
        cum_proportions = np.cumsum(proportions)
        return proportions, cum_proportions


    def principal_components(self):
        eig_vals, eig_vecs = self.eig_vals_vecs()
        principal_vals = eig_vals[:self.n_components]
        principal_components = eig_vecs[:, :self.n_components]
        return principal_components

    
    def transform(self):
        p_comps = self.principal_components()
        stand_data = self.standardize_data()
        return stand_data @ p_comps
    
    def plot(self, x, y):
        plt.scatter(x[:,0], x[:,1], c=y)
        plt.show()


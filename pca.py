#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class myPCA:
    
    def __init__(self, n_components):
        self.fitted=False
        self.n_components = n_components
        self.trans= []
        self.variance_ratio = []
        self.variance = []
    def fit(self,x):
        Mean = np.mean(x,axis=0)
        Variance = np.cov(x.T)

        vals, vecs = np.linalg.eig(Variance)
        sum_values = np.sum(vals)
        var_ratios = vals/sum_values
        y = vecs.T.dot(x.T)
        yt = y.T

        for i in range(yt.shape[0]):
            for j in range(yt.shape[1]):
                if(j%2!=0):
                    yt[i][j] = -yt[i][j]
        self.fitted=True
        self.trans=yt
        li = [i for i in range(self.n_components)]
        self.variance_ratio = var_ratios
        self.variance_ratio = self.variance_ratio[:self.n_components]
        self.variance = vals
        self.variance = self.variance[:self.n_components]
    def transform(self):
        if(self.fitted==False):
            print("Data not fitted");
        li = [i for i in range(self.n_components)]
        return (self.trans[:,li])


# In[ ]:





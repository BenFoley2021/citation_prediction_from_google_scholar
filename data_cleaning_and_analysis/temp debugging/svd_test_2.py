# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:31:56 2021
https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783989485/1/ch01lvl1sec21/using-truncated-svd-to-reduce-dimensionality
@author: Ben
"""

from sklearn.datasets import load_iris
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(2)
iris_transformed = svd.fit_transform(iris_data)
iris_data[:5]

iris_transformed[:5]
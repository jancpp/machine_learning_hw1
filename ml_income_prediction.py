#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file:   ml_income_prediction.py
@date:   2/6/2019
@author: Jan Polzer and Ryan Duckworth
"""

import sys
import numpy 
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}' .format(sys.version))
print('Numpy: {}' .format(numpy.__version__))
print('Pandas: {}' .format(pandas.__version__))
print('Matplotlib: {}' .format(matplotlib.__version__))
print('Seaborn: {}' .format(seaborn.__version__))
print('Scipy: {}' .format(scipy.__version__))
print('Sklearn: {}' .format(sklearn.__version__))

# import packeges
import numpy as np              # math tools
import pandas as pd             # import and manage datasets
import matplotlib.pyplot as plt # plot charts

# load dataset
data = pd.read_csv('adult.csv')
X = data.iloc[:, :-1].values # matrix of independent variables
y = data.iloc[:, 14].values  # dependent variable vector

# split dataset into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# explore dataset
print(data.columns)
print(data.shape)
print(data.describe())

data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)

# plot historgram for each parameter
data.hist(figsize = (20, 20))
plt.show()

# determine if income is over/uner $50k
Over = data[data['income'] == '<=50K']
Under = data[data['income'] == '>50K']

print('Income over $50k: {}'.format(len(Over)) + ' samples')
print('Income under $50k: {}'.format(len(Under)) + ' samples')


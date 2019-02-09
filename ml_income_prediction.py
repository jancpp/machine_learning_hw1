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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
data = pd.read_csv('adult.csv')

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
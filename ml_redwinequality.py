#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EECS 738 - Machine learning
Dataset from https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009#winequality-red.csv
Due date: 2/18/2019
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
data = pd.read_csv('winequality-red.csv')



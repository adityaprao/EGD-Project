import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

filepath = 'raw_values.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
print(df.shape)
df.tail()
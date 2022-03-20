import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/prostate.csv')
df.corr() # many strong correlations (> .5)

# Unit variance
standardize = lambda x: ((x - np.mean(x)) / np.std(x))

X_train = standardize(df.query("train == 'T'").drop(columns=['lpsa', 'train']))
y_train = standardize(df.query("train == 'T'")['lpsa'])
X_test = standardize(df.query("train == 'F'").drop(columns=['lpsa', 'train']))
y_test = standardize(df.query("train == 'F'")['lpsa'])

# Fit using all predictors
model = LinearRegression().fit(X_train, y_train)
y_hat = model.predict(X_test)
sigma_hat = (1.0 / (len(X_train) - X_train.shape[1] - 1)) * np.sum((y_test - y_hat)**2)

# Calculate Z-score (Eq. (3.12))
V = np.diag(np.linalg.inv(X_train.T @ X_train))
Z = model.coef_ / (sigma_hat * np.sqrt(V))

# A Z-score greater than 2 in absolute value is approx. significant at the 5% level
X_train_small = X_train.iloc[:, Z > 2]
X_test_small = X_test.iloc[:, Z > 2]

# Fit model with subset of predictors
model_small = LinearRegression().fit(X_train_small, y_train)
y_hat_small = model_small.predict(X_test_small)

# Calculate RSS for both models
rss_big_model = ((y_hat - y_test)**2).sum()
rss_small_model = ((y_hat_small - y_test)**2).sum()

# Eq (3.13), p. 67 ESL
F_statistics = (
  ((rss_small_model - rss_big_model) / (X_train.shape[1] - X_train_small.shape[1])) /
  rss_big_model / (len(X_train) - X_train.shape[1] - 1)
)

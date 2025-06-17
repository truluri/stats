#!/usr/bin/env python
# coding: utf-8
from IPython.display import display

# # Finans Projekt 1 - English
# This notebook is for assistance with the coding for many of the questions in the project.
# The sections are marked with the corresponding question in the Project description.
# Remember, this code is provided to get started with the project, but the code is not complete for answering the corresponding questions

# #### Initialize python packages

# In[24]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm


# #### Read Data

# In[25]:


# path to project data (replace with your own path)
file_path = "./finans1_data.csv"
## Read data into a pandas DataFrame
D = pd.read_csv(file_path, delimiter=";")
D[["t", "AGG", "VAW", "IWN", "SPY"]]


# #### a) Simple summary of data

# In[26]:


D = D[["t", "AGG", "VAW", "IWN", "SPY"]]
print(
    f"Dimension of DataFrame: {D.shape}"
)  # f-strings allow us to insert variables directly into the string
print(f"Variable names: {D.columns}")
# print("\nFirst few rows of DataFrame:") # \n is the newline character for strings
display(D.head())
# print("Last row of DataFrame:")
display(D.tail())
# print("Some summary statistics:")
display(D.describe())
# print("Data types:", D.dtypes)


# #### b) Summary statistics

# In[27]:


# Calculate the required statistics
D2 = D[["AGG", "VAW", "IWN", "SPY"]]
no_obs = D2.notna().sum()
avg = D2.mean()
var = D2.var(ddof=1)
sd = D2.std(ddof=1)
quantiles = pd.DataFrame(
    np.quantile(D2, [0.25, 0.5, 0.75], method="averaged_inverted_cdf", axis=0).T,
    index=D2.columns,
)
# print(D2)


# In[ ]:


# #### Statistical analysis I

# #### d)

# In[28]:


covariance_table = D[["AGG", "VAW", "IWN", "SPY"]].cov()
print(covariance_table)


# #### f)

# #### Problem 2 - Best investment

# #### g)

# In[29]:


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sm.qqplot(D2["AGG"].dropna(), line="q", a=1 / 2, ax=axs[0, 0])
axs[0, 0].set_title("AGG")
## Do the same for the other ETFs


# #### h) Tests

# In[30]:


###########################
## Calculations of the 95% confidence intervals
## t-quantile for the confidence interval for the mean of AGG,
## since the degrees of freedom for the mean of AGG are 453
cri_val = stats.t.ppf(0.975, 453)
cri_val


# ## EXTRA
# #### Subsets in Python

# In[31]:


## df['AGG'] < 0 returns all observations where AGG is negative
## Can be used to extract all AGG losses
loss_weeks = D["AGG"] < 0
agg_losses = D["AGG"][loss_weeks]
print("Weeks with negative returns in AGG:")
# display(agg_losses)

## Alternatively, use the 'query' method
agg_losses_query = D.query("AGG < 0")
print("Weeks with negative returns in AGG (query method):")
# display(agg_losses_query)
# Or use the 'loc' method
agg_losses_loc = D.loc[D["AGG"] < 0, "AGG"]
print("Weeks with negative returns in AGG (loc method):")
# display(agg_losses_loc)

## More complex logical expressions can be made, e.g.:
## Find all observations from weeks where AGG had a loss and SPY had a gain
agg_loss_spy_gain = D.query("AGG < 0 & SPY > 0")
print("Weeks with negative AGG returns and positive SPY returns:")
# display(agg_loss_spy_gain)

# "display()" function gives a nicer table than print. It is
# especially useful when working with dataframes (pandas)


# #### Additional Python tips

# In[32]:


## Make a for loop to calculate some summary
## statistics and save the result in a new data frame
Tbl = pd.DataFrame()
for i in ["AGG", "VAW", "IWN", "SPY"]:
    Tbl.loc[i, "ETF_mean"] = D[i].mean()
    Tbl.loc[i, "ETF_var"] = D[i].var(ddof=1)

# show
# display(Tbl)


# In[33]:


# There are many other ways to do these calculations, some more concise. For example
# Calculate mean and variance for all columns but 't'
result = D.drop(columns="t").agg(["mean", "var"])

# The agg function(aggregate) is used to calculate the mean and variance of returns for each ETF.
# display(result)

# See more functions in pandas documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# Numpy documentationen: https://numpy.org/doc/stable/reference/index.html
# Or find documentation or guides on other python packages/functions online.


# #### Latex Tips
# Pandas (pd) also includes a function that is very handy for writing tables/dataframes directly into Latex-code.
# This is done by usind the function `pd.to_latex()`.

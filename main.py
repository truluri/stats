#!/usr/bin/env python
# coding: utf-8


# # Finans Projekt 1 - English
# This notebook is for assistance with the coding for many of the questions in the project.
# The sections are marked with the corresponding question in the Project description.
# Remember, this code is provided to get started with the project, but the code is not complete for answering the corresponding questions


from IPython.display import display
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Create a new cell in your notebook and run this code
etfs = ["AGG", "VAW", "IWN", "SPY"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot Empirical Densities (Histograms with density=True)

# #### Read Data

file_path = "./finans_project/finans1_data.csv"
## Read data into a pandas DataFrame
D = pd.read_csv(file_path, delimiter=";")
D[["t", "AGG", "VAW", "IWN", "SPY"]]
# Convert to percentages and plot
D_percent = D[etfs]
D_percent.plot(
    kind="density",
    subplots=True,
    layout=(2, 2),
    ax=axes.flatten()[:4],
    sharex=False,
    title="Empirical Density of Weekly Returns",
)
# --- Code for Summary Table ---

# Get the full descriptive statistics
desc_stats = D[["AGG", "VAW", "IWN", "SPY"]].describe()

# Get the sample variance (ddof=1 is the default for pandas, but it's good to be explicit)
variance = D[["AGG", "VAW", "IWN", "SPY"]].var(ddof=1)

# Create a new DataFrame to hold the results in the desired order
summary_table = pd.DataFrame(
    {
        "Number of obs.": desc_stats.loc["count"],
        "Sample mean": desc_stats.loc["mean"],
        "Sample variance": variance,
        "Std. dev.": desc_stats.loc["std"],
        "Lower quartile": desc_stats.loc["25%"],
        "Median": desc_stats.loc["50%"],
        "Upper quartile": desc_stats.loc["75%"],
    }
).astype(float)  # Ensure all values are floats for consistent formatting

# Display the final table
print("Summary Statistics Table:")
display(summary_table)

# --- Code for Plots ---

etfs = ["AGG", "VAW", "IWN", "SPY"]

# 1. Empirical Density Plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
D[etfs].plot(
    kind="density", subplots=True, layout=(2, 2), ax=axes.flatten(), sharex=False
)
fig.suptitle("Empirical Density of Weekly ETF Returns", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

# Save the figure to a file
fig.savefig("bempirical_density_plots.png", dpi=300)  # dpi=300 for high quality
plt.show()


# 2. Box Plots
fig, ax = plt.subplots(figsize=(10, 7))
D[etfs].plot(kind="box", ax=ax)
ax.set_title("Box Plots of Weekly ETF Returns", fontsize=16)
ax.set_ylabel("Weekly Return")
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# Save the figure to a file
fig.savefig("bbox_plots.png", dpi=300)
plt.show()
# #### a) Simple summary of data


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

# Calculate summary statistics and export to CSV
D2 = D[["AGG", "VAW", "IWN", "SPY"]]
no_obs = D2.notna().sum()
avg = D2.mean()
var = D2.var(ddof=1)
sd = D2.std(ddof=1)
quantiles = pd.DataFrame(
    np.quantile(D2, [0.25, 0.5, 0.75], method="averaged_inverted_cdf", axis=0).T,
    index=D2.columns,
    columns=["Q25", "Q50", "Q75"],
)

# Create summary statistics DataFrame
summary_stats = pd.DataFrame(
    {
        "No_Obs": no_obs,
        "Mean": avg,
        "Variance": var,
        "Std_Dev": sd,
        "Q25": quantiles["Q25"],
        "Q50": quantiles["Q50"],
        "Q75": quantiles["Q75"],
    }
)
result = D.drop(columns="t").agg(["mean", "var"])
print(result)
# Export to CSV

summary_stats.to_csv("summary_statistics.csv")
print("Summary statistics exported to summary_statistics.csv")
print(summary_stats)


# In[ ]:


# #### Statistical analysis I

# #### d)

# In[28]:

print("d")
covariance_table = D[["AGG", "VAW", "IWN", "SPY"]].cov()
correlation_table = D[["AGG", "VAW", "IWN", "SPY"]].corr()
print(covariance_table)
print(correlation_table)

# Export covariance table to CSV
covariance_table.to_csv("covariance_table.csv")
print("Covariance table exported to covariance_table.csv")
correlation_table.to_csv("correlation_table.csv")
print("Correlation table exported to covariance_table.csv")


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

print(D2.shape)


def main():
    print("Hello from stats!")


if __name__ == "__main__":
    main()

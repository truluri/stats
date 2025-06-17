#!/usr/bin/env python
# coding: utf-8

# # Introduction to mathematical statistics
#
# Welcome to the lecture 7 in 02403
#
# During the lectures we will present both slides and notebooks.
#
#

# In[138]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm


# ## Example: Skive fjord

# In[139]:


SkiveAvg = pd.read_csv("skiveAvg.csv", sep=";")
chla = np.array(SkiveAvg["chla"])
stats.ttest_1samp(chla, popmean=0).confidence_interval(0.95)


# Check assumptions

# In[ ]:


sm.qqplot(chla, line="q", a=1 / 2)
plt.tight_layout()
plt.show()


# Normal assumption clearly violated, check log-data

# In[ ]:


sm.qqplot(np.log(chla), line="q", a=1 / 2)
plt.tight_layout()
plt.show()


# looks good, how about independence assumption

# In[ ]:


# lchla = np.log(SkiveAvg["chla"])
lchla = np.log(chla)
n = len(chla)
lchla1 = lchla[1 : (n - 1)]
lchla2 = np.roll(lchla, -1)[1 : (n - 1)]

np.corrcoef(lchla1, lchla2)[0, 1]


# In[ ]:


plt.scatter(lchla1, lchla2)


# In[ ]:


plt.plot(lchla)


# Independece assumption clearly violated, and more advanced models are needed. Conclusion might be wrong!

# ## Example: Tensile strength

# Conf. int.

# In[ ]:


s1 = 54.24
s2 = 28.54
y1_bar = 1250
y2_bar = 1300
n1 = 25000
n2 = 15000
sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
ME = stats.t.ppf(0.975, df=n1 + n2 - 2) * sp * np.sqrt(1 / n1 + 1 / n2)
D_bar = y1_bar - y2_bar
print("Conf. Int. diff", [D_bar - ME, D_bar + ME])


# Hypothesis test

# In[ ]:


delta0 = -50
t_obs = (D_bar - delta0) / (sp * np.sqrt(1 / n1 + 1 / n2))
print("test statistics", t_obs)
print("Critical value", stats.t.ppf(0.975, df=n1 + n2 - 2))
print("pvalue", 2 * (1 - stats.t.cdf(t_obs, df=n1 + n2 - 2)))


# Is the assumption reasonable?

# In[ ]:


F_obs = s1**2 / s2**2
1 - stats.f.cdf(F_obs, n1 - 2, n2 - 1)


# So no the assumptions are not reasonable

# ## Example: Tensile strength 2

# In[ ]:


s1 = 54.24
s2 = 28.54
y1_bar = 1250
y2_bar = 1300
n1 = 25000
n2 = 15000
v = (s1**2 / n1 + s2**2 / n2) ** 2 / (
    (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
)
MEw = stats.t.ppf(0.975, df=v) * np.sqrt(s1**2 / n1 + s2**2 / n2)
print("Welch Conf. Int. diff", np.array([D_bar - MEw, D_bar + MEw]))
print("Conf. Int. diff", np.array([D_bar - ME, D_bar + ME]))

print("pooled df vs Welch df", [n1 + n2 - 2, v])


# ## Illustrating Welch df

# In[ ]:
# %%

n1 = 6
n2 = 6
s1 = 1
s2 = np.arange(0, 10, 0.01)
v = (s1**2 / n1 + s2**2 / n2) ** 2 / (
    (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
)
print("welch")
plt.plot(s2, v)
# %%

# ## Overlapping CI

# In[ ]:


s1 = 4 / (2 * 1.96) * np.sqrt(100)
s2 = 4 / (2 * 1.96) * np.sqrt(100)
y1_bar = 2
y2_bar = 5
sp = np.sqrt(((s1**2 + s2**2) / 2))
ME = 1.96 * sp * np.sqrt(2 / 100)

Ci_low = 3 - ME
Ci_high = 3 + ME
print(np.array([Ci_low, Ci_high]))


# ## Skive Fjord 1

# In[ ]:


SkiveAvg = pd.read_csv("skiveAvg.csv", sep=";")
print(SkiveAvg)
temp = SkiveAvg["temp"]
month = SkiveAvg["month"]
year = SkiveAvg["year"]
temp = temp[month == 7]
temp1 = temp[year < 1995]
temp2 = temp[year >= 1995]


# In[ ]:


test = stats.ttest_ind(temp1, temp2, equal_var=True)
test.confidence_interval(0.95)


# In[ ]:


n1 = len(temp1)
n2 = len(temp2)
X = np.array(
    [np.repeat(1, n1 + n2), np.append(0.5 * np.repeat(1, n1), -0.5 * np.repeat(1, n2))]
).T
XXI = np.linalg.inv(X.T @ X)
beta = XXI @ X.T @ temp
H = X @ XXI @ X.T
I = np.identity(n1 + n2)
sig_hat = np.sqrt(temp.T @ (I - H) @ temp / (n1 + n2 - 2))
se_beta = sig_hat * np.sqrt(np.diag(XXI))

tq = stats.t.ppf(0.975, df=n1 + n2 - 2)
Ci_low = beta - tq * se_beta
Ci_high = beta + tq * se_beta
coefTab = np.array([beta, se_beta, Ci_low, Ci_high]).T
coefTab
col_names = ["Estimates", "Std.Error", "CI_low", "CI_high"]
row_names = ["beta0", "beta1"]
coefTab = pd.DataFrame(coefTab, columns=col_names, index=row_names)
print(np.mean(temp2))
coefTab


# In[ ]:


r = (I - H) @ temp
h = np.diag(H)
r_tilde = r
r_tilde[0 : (n1 - 1)] = r_tilde[0 : (n1 - 1)] / np.sqrt(h[0 : (n1 - 1)])
r_tilde[n1 : (n1 + n2 - 1)] = r_tilde[n1 : (n1 + n2 - 1)] / np.sqrt(
    h[n1 : (n1 + n2 - 1)]
)
# r_stan = pd.DataFrame({    "Group 1": r_tilde[0:(n1-1)],"Group 2": np.append(r_tilde[n1:(n1+n2-2)],np.nan) })
r_stan = pd.DataFrame(
    {
        "Group 1": r_tilde[0 : (n1 - 1)],
        "Group 2": np.append(r_tilde[n1 : (n1 + n2 - 1)], np.nan),
    }
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
r_stan.boxplot(ax=ax1)
sm.qqplot(r_tilde, line="q", a=1 / 2, ax=ax2)
plt.tight_layout()
plt.show()


# ## Skive fjord II

# Test using build in functions

# In[ ]:


Nin = SkiveAvg["N.load"]
year = SkiveAvg["year"]
Nin1999 = np.array(Nin[year == 1999])
Nin2006 = np.array(Nin[year == 2006])
Nin_test = stats.ttest_ind(Nin1999, Nin2006, equal_var=False)
Nin_test.pvalue


# So no we would accept no difference, but look at data

# In[ ]:


plt.plot(Nin1999)
plt.plot(Nin2006)


# Clear yearly variation

# In[ ]:


Nin_test_pair = stats.ttest_rel(Nin1999, Nin2006)
print(Nin_test_pair.confidence_interval(0.95))
## Or
Nin_test_pair2 = stats.ttest_1samp(Nin1999 - Nin2006, popmean=0)
print(Nin_test_pair2.confidence_interval(0.95))


# ## Power and sample size in Python

# In[ ]:


# The sample size for power=0.80
import statsmodels.stats.power as smp

delta = 4
sd = 12
alpha = 0.05
power = 0.8
smp.TTestPower().solve_power(effect_size=delta / sd, alpha=alpha, power=power)


# In[ ]:


# Finding the sample size for detecting a group difference of 2
# with sigma=1 and power=0.9
delta = 2
sd = 1
alpha = 0.05
power = 0.90
smp.TTestIndPower().solve_power(
    effect_size=delta / sd, alpha=alpha, power=power, ratio=1
)

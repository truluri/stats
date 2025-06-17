#!/usr/bin/env python
# coding: utf-8

# # Introduction to mathematical statistics 
# 
# Welcome to the lecture 5 in 02403
# 
# During the lectures we will present both slides and notebooks. 
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# ## Example: Probability region

# In[ ]:


Sigma = np.array([[1,1],[1,2]])
print(stats.chi2.ppf(0.95,2))
print("Sigma = ",Sigma)
print("Sigma^-1= ",np.linalg.inv(Sigma))


# Probability region

# In[ ]:


y =  np.arange(-np.sqrt(6),np.sqrt(6),0.001)
plt.plot(y,y+np.sqrt(6-y**2))
plt.plot(y,y-np.sqrt(6-y**2))


# Add eigen vectors

# In[ ]:


Lambda, V = np.linalg.eig(Sigma)

print(V)
plt.plot(y,y+np.sqrt(6-y**2),scalex=[-3,3])
plt.plot(y,y-np.sqrt(6-y**2))
plt.plot([0,V[0,0]*np.sqrt(Lambda[0]*6)],[0,V[1,0]*np.sqrt(Lambda[0]*6)])
plt.plot([0,-V[0,1]*np.sqrt(Lambda[1]*6)],[0,-V[1,1]*np.sqrt(Lambda[1]*6)])
plt.axis([-3.5,3.5,-3.5,3.5])



# ## Example: projecton mat

# In[ ]:


A = np.array([[1/2,-1/2],[-1/2,1/2]])
I = np.array([[1,0],[0,1]])
print(I-A)
print(A)

print(A@A)


# ## Items on a scale

# In[ ]:


X1 = np.array([[1,0],[0,1],[1,1]])
print(np.linalg.inv(X1.T@X1)@X1.T)

X2 = np.array([[1,0],[1,1],[2,1]])
print(np.linalg.inv(X2.T@X2)@X2.T)

X3 = np.array([[1,1],[1,-1],[2,0]])
print(np.linalg.inv(X3.T@X3)@X3.T)


# Items on a scale orthogonal?

# In[ ]:


print(X1.T@X1)
print(X2.T@X2)
print(X3.T@X3)


# ## Items on a scale

# In[ ]:


H1 = X1 @ np.linalg.inv(X1.T@X1)@X1.T
H2 = X2 @ np.linalg.inv(X2.T@X2)@X2.T
H3 = X3 @ np.linalg.inv(X3.T@X3)@X3.T

print(np.max(np.abs(H1-H2)))
print(np.max(np.abs(H1-H3)))
H1


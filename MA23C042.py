#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT 01

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix


# In[5]:


# Load csv file into pandas Data Frame
df=pd.read_csv("Planet.csv")
df


# ## Data Cleaning

# In[7]:


df.columns=['x','y']


# In[8]:


sns.scatterplot(data=df, x='y',y='x')


# In[9]:


df.describe()


# In[10]:


df.head(120)


# In[11]:


df.tail(250)


# In[12]:


df.info()


# In[13]:


df['x'] = df['x'].round(2)
df['y'] = df['y'].round(2)


# In[14]:


pd.Series(df['x'].unique())


# In[17]:


# Cleaned dataframe
df


# In[18]:


from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(df)
print(sparse_matrix)
sparse_matrix.ndim
print(sparse_matrix.dtype)
print(type(sparse_matrix))


# In[19]:


arr= sparse_matrix.toarray()
print(arr)
print(arr.ndim)
print(arr.shape)
print(type(arr))


# In[20]:


arr = sparse_matrix.toarray()
print(arr)
print(arr.ndim)
print(arr.shape)
print(type(arr))
import numpy as np
from scipy.sparse import csr_matrix

datapoints = 1000

min_x = np.min(arr[:, 0])
min_y = np.min(arr[:, 1])
max_x = np.max(arr[:, 0])
max_y = np.max(arr[:, 1])

# we defined min and max as it is need in discretization.

#IMPORTANT as formula i.e[ normalization * (n-1)]
discretized_x = np.clip(((arr[:, 0] - min_x) / (max_x - min_x) * (datapoints- 1)).astype(int), 0, datapoints - 1)
discretized_y = np.clip(((arr[:, 1] - min_y) / (max_y - min_y) * (datapoints - 1)).astype(int), 0, datapoints- 1)

print(discretized_x.ndim)  
print(discretized_x.size)  

boolean_m = np.zeros((datapoints, datapoints), dtype=bool)  

for x, y in zip(discretized_x, discretized_y): 
    boolean_m[x, y] = True   
        
sparse_matrix1 = csr_matrix(boolean_m)
print(sparse_matrix1)
print(sparse_matrix1.ndim)
print(type(sparse_matrix1))


# In[21]:


num_1 = sparse_matrix1.toarray()
print(num_1)
print(num_1.ndim)
print(num_1.size)


# In[22]:


# Rotate by 90 degree
rotate_df1 = np.rot90(num_1, k=-1)
rotate_df1


# In[23]:


# FLip 
rotate_df2 = np.rot90(num_1, k=-2)
rotate_df2


# In[24]:


# 2nd Image

import matplotlib.pyplot as plt
import numpy as np

rows, cols = np.nonzero(rotate_df1)

# Create a subplot and scatter plot
plt.figure(figsize=(10, 10))
ax1 = plt.subplot()
ax1.scatter(rows, cols , s=30)
plt.grid(True)

# Display the plot
plt.show()


# In[25]:


# 3rd Image

import matplotlib.pyplot as plt
import numpy as np

rows, cols = np.nonzero(rotate_df2)

# Create a subplot and scatter plot
plt.figure(figsize=(8, 8))
ax1 = plt.subplot()
ax1.scatter(rows, cols , s=70)
plt.grid(True)

# Display the plot
plt.show()


# In[ ]:





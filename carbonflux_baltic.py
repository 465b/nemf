#!/usr/bin/env python
# coding: utf-8

# # Carbon Fluxes within the Baltic Sea ecosystem

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ### importing the initial values from file 

# In[ ]:


"""
all files are in plain text and tab seperated
d0 represents the actual carbon mass values. d0 because it is the zeroth derivation.
d1 is the carbon mass flux
d2 is the change of the carbon mass flux
"""

d0_init = np.genfromtxt("d0_init",delimiter='\t', skip_header=1)[:,1:]  #,names=True
d1_init = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
d2_init = np.genfromtxt("d2_init", delimiter='\t', skip_header=1)[:,1:]
names = ["PHY","MAC","PZO","ZOO","BEN","FLO","PLA","HER","SPR","COD","MMA","BAC","POC","DOC","SED"]


# ### defining worker functions

# In[ ]:


def runge_kutta(d0,d1,d2,d0_reset,time_step,time_step_size):
    pass


def euler_forward(d0,d1,d2,d0_reset,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    time = time_step * time_step_size
    d0 = d0 + d1*time_step_size
    d0 = d0_exec_reset(d0, d0_reset,time)
    d1 = d1 + np.matmul(d2,d1)*time_step_size
    time_step += 1
    
    return d0, d1, time_step
    
    
def d0_exec_reset(d0, d0_reset,time):
    """ the carbon masses are resetet with a certain frequency
        this function checks if its time to reset the current volume and does so """
    for ii,reset_list in enumerate(np.transpose(d0_reset)):
        reset_freq = 1/reset_list[1]
        time_postreset = time%(reset_freq)
        #print(time_postreset, time, reset_freq)
        if (time_postreset < time_step_size):
            #print("reset " + str(ii))
            d0[ii] = reset_list[0]
            
    return d0    


# In[ ]:


# renaming initial values for better readability
d0_reset = d0_init
d1 = d1_init
d2_weights = d2_init

# calculate all time constants
lowest_freq = max(1/d0_reset[1])
n_obs = len(d1)

max_time = 100 #years
time_step_size = 1/(365)
n_steps = int(100/time_step_size)
#n_steps = int(lowest_freq * 365 * 1)


# initialize data arrays
d0_log = np.zeros( (n_steps,n_obs) )
d1_log = np.zeros( (n_steps,n_obs) )
d0_log[0] = d0_reset[0]
d1_log[0] = d1

# calculate the time evolution
time_step = 0
for ii in np.arange(1,n_steps):
    d0_log[ii],d1_log[ii],time_step = euler_forward(d0_log[ii-1],d1_log[ii-1],d2_weights,d0_reset,
                                              time_step,time_step_size)   


# In[ ]:


# plotting
time = np.arange(np.shape(d0_log)[0]) * time_step_size 

plt.figure(figsize=(10,5))
for kk in np.arange(len(d1_log[0])):
    plt.title( "Absolute Carbon Mass")
    plt.plot(time,d0_log[:,kk],label=names[kk])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Absolute Carbon mass [mg]")
    plt.xlabel("Time [years]")
    
plt.figure(figsize=(10,5))
for kk in np.arange(len(d1_log[0])):
    plt.title( "Carbon Mass Flux")
    plt.plot(time,d1_log[:,kk],label=names[kk])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Carbon mass change [mg per yer]")
    plt.xlabel("Time [years]")


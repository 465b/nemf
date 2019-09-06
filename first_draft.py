#!/usr/bin/env python
# coding: utf-8

# # Carbon Fluxes within the Baltic Sea ecosystem

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


d0_init = np.genfromtxt("d0_init",delimiter='\t', skip_header=1)[:,1:]  #,names=True
d1_init = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
d2_init = np.genfromtxt("d2_init", delimiter='\t', skip_header=1)[:,1:]
names = ["PHY","MAC","PZO","ZOO","BEN","FLO","PLA","HER","SPR","COD","MMA","BAC","POC","DOC","SED"]


# In[ ]:


def recalculate_d1(d1,d2):
    return np.matmul(d2,d1)

def stepper(d0,d1,d2,d0_reset,time_step,time_step_size):
    time = time_step * time_step_size
    d0 = d0 + d1
    d1 = recalculate_d1(d1,d2)
    
    """ checks if its time to reset current volume and does so """
    for ii,reset_list in enumerate(np.transpose(d0_reset)):
        reset_freq = 1/reset_list[1]
        time_postreset = time%(reset_freq)
        #print(time_postreset, time, reset_freq)
        if (time_postreset < time_step_size):
            #print("reset " + str(ii))
            d0[ii] = reset_list[0]
    
    time_step += 1
    
    #print("d0 pre return {}".format(d0))
    return d0, d1, time_step


# In[ ]:


d0_reset = d0_init
d1 = d1_init
d2_weights = d2_init


lowest_freq = max(1/d0_reset[1]) # read in from file
n_obs = len(d1)
n_steps = 20 #int(lowest_freq * 365 * 0.01)
time_step_size = 1/365

d0_log = np.zeros( (n_steps,n_obs) )
d1_log = np.zeros( (n_steps,n_obs) )
d2_log = np.zeros( (n_steps,n_obs, n_obs) )
d0_log[0] = d0_reset[0]
d1_log[0] = d1


time_step = 0
for ii in np.arange(1,n_steps):
    d0_log[ii],d1_log[ii],time_step = stepper(d0_log[ii-1],d1_log[ii-1],d2_weights,d0_reset,
                                              time_step,time_step_size)


# In[ ]:


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
    


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

d0_init = np.genfromtxt("d0_init",delimiter='\t', skip_header=1)[1:]  #,names=True
d1_init = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
d2_init = np.genfromtxt("d2_init", delimiter='\t', skip_header=1)[:,1:]
names = ["PHY","MAC","PZO","ZOO","BEN","FLO","PLA","HER","SPR","COD","MMA","BAC","POC","DOC","SED"]

random_seed = 137


# ### defining worker functions

# In[ ]:


def runge_kutta(d0,d1,d2_weights,d0_reset,time_step,time_step_size):
    pass


def euler_forward(d0,d1,d1_weights,d2_weights,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    time = time_step * time_step_size
    d0 = d0 + d1*time_step_size
    #d0 = d0_exec_reset(d0, d0_reset,time)
    d1 = d1 + np.matmul(d2_weights,d1)*time_step_size
    time_step += 1
    
    return d0, d1, time_step


def normalize_columns(d2):
    """ enforces that all columns of d2 are normalized to 1
        otherwise an carbon mass transport amplification would occur """
    overshoot = np.sum(d2,axis=0)
    n_obs = len(d2_weights)
    for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
        if truth_val : 
            if (d2[ii,ii] == 0) :
                # not normalized because its by choice
                # designed to be an infinite source
                pass
            elif d2[ii,ii] == overshoot[ii]:
                # only the diagnoal entry is filled
                # hence, its the dump. 
                pass 
            else: 
                # the following assumed that only diagonal values can be negative
                buffer = d2[ii,ii]
                d2[:,ii] -= d2[:,ii]/(overshoot[ii]-buffer)*overshoot[ii]
                d2[ii,ii] = buffer
    return d2


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


def interaction_matrix_modifier():
    """ this function is inteded to modify the interactino matrices (d1_/d2_weights)
        to find appropriate solutions to the system. """
    
    """ we have 53 non-zero, non-negative-one values.
        len(d2_weights[((d2_weights != -1) & (d2_weights != 0)) == True])
        
        assuming we constrain our precision to a resolution of 5% 
        we have a parameter searchspace of 20**53
        which gives us convenient search time of 3e61 years
        wow, such convenience """


def run_time_evo(method,max_time,time_step_size,d0,d1,d1_weights,d2_weights):
    """ run_time_evo(method, max_time, time_step_size = 1/(365),d0=d0,d1=d1,d2_weights=d2_weights) """
    
    # calculate all time constants
    n_obs = len(d1)
    n_steps = int(max_time/time_step_size)
    
    # initialize data arrays
    d0_log = np.zeros( (n_steps,n_obs) )
    d1_log = np.zeros( (n_steps,n_obs) )
    d0_log[0] = d0[0]
    d1_log[0] = d1

    # calculate the time evolution
    time_step = 0
    for ii in np.arange(1,n_steps):
        d0_log[ii],d1_log[ii],time_step = method(d0_log[ii-1],d1_log[ii-1],d1_weights,d2_weights,
                                                  time_step,time_step_size)
    
    return d0_log, d1_log, time_step_size


def plotting_coupling_matrix(d2_weights,d1_weights):
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title("d2 coupling matrix")
    plt.imshow(d2_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()

    ax = plt.subplot(122)
    ax.set_title("d1 coupling matrix")
    plt.imshow(d1_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()
    plt.show()


def plotting_standard(d0_log,d1_log,time_step_size,names):
    """ plotting_standard(d0_log=d0_log,d1_log=d1_log,time_step_size=time_step_size,names=names) """
    # plotting
    time = np.arange(np.shape(d0_log)[0]) * time_step_size 

    plt.figure(figsize=(20,5))
    plt.subplot(121)
    for kk in np.arange(len(d1_log[0])):
        plt.title( "Absolute Carbon Mass")
        plt.plot(time,d0_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Absolute Carbon mass [mg]")
        plt.xlabel("Time [years]")
    

    plt.subplot(122)
    for kk in np.arange(len(d1_log[0])):
        plt.title( "Carbon Mass Flux")
        plt.plot(time,d1_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Carbon mass change [mg per yer]")
        plt.xlabel("Time [years]")
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(20,5))
    
    plt.subplot(121)
    plt.plot(time, np.sum(d0_log,axis = 1) - 2*d0_log[:,-1])
    plt.title("Total Carbon mass over time")
    plt.ylabel("Total Carbon mass [mg]")
    plt.xlabel('Time [years]')

    plt.subplot(122)
    plt.plot(time, np.sum(d1_log,axis = 1) - 2*d1_log[:,-1] )
    plt.title("Total Carbon mass flux over time")
    plt.ylabel("Total Carbon mass flux [mg/y]")
    plt.xlabel('Time [years]')
    plt.tight_layout()
    plt.show()


# ### running basic sim

# In[ ]:


# renaming initial values for better readability
d0 = d0_init.copy()
d1 = d1_init.copy()
d2_weights = d2_init.copy()
d2_weights = normalize_columns(d2_weights)
d1_weights = d2_weights*0
tss = 1/365

d0_log,d1_log,time_step_size = run_time_evo(euler_forward, 50, tss,
                                           d0,d1,d1_weights,d2_weights)
plotting_standard(d0_log,d1_log,tss,names)


# ### optimizer to find an arbitrary flow balance

# In[ ]:


from scipy.optimize import root


# In[ ]:


def find_root(x):
    d0 = d0_init.copy()
    d1 = d1_init.copy()
    d2 = d2_init.copy()
    d2[-1] += x
    d2 = normalize_columns(d2)
    
    d0_log,d1_log,time_step_size = run_time_evo(euler_forward, 50, tss,
                                           d0,d1,d1_weights,d2)

        
    return (np.sum(d1_log,axis = 1) - 2*d1_log[:,-1])[-1]

root_out = root(find_root, 0.56)


d0 = d0_init.copy()
d1 = d1_init.copy()
d2 = d2_init.copy()
d2[-1] += root_out['x'][0]
d2_weights = normalize_columns(d2)

d0_log,d1_log,time_step_size = run_time_evo(euler_forward, 50, tss,
                                       d0,d1,d1_weights,d2_weights)

plotting_standard(d0_log,d1_log,tss,names)


# ### adding a dependency for carbon mass

# In[ ]:


def euler_forward_extended(d0,d1,d1_weights, d2_weights,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    time = time_step * time_step_size
    d0 = d0 + d1*time_step_size
    #d0 = d0_exec_reset(d0, d0_reset,time)
    d1 = d1 + ( np.matmul(d2_weights,d1) + np.matmul(d1_weights,d0) )*time_step_size
    time_step += 1
    
    return d0, d1, time_step


# In[ ]:


# renaming initial values for better readability
d0 = d0_init.copy()
d1 = d1_init.copy()
d2_weights = d2_init.copy()
d2_weights = normalize_columns(d2_weights)

np.random.seed(random_seed)
d1_weights = d2_weights *np.random.rand(np.shape(d2_weights)[0],np.shape(d2_weights)[0])
for ii in np.arange(2,len(d1_weights)):
    d1_weights[ii,ii] = -1
d1_weights = normalize_columns(d1_weights)


d0_log,d1_log,time_step_size = run_time_evo(euler_forward_extended,max_time=100,time_step_size=1/(365),
                                           d0=d0,d1=d1,d1_weights=d1_weights,d2_weights=d2_weights)
plotting_standard(d0_log,d1_log,time_step_size,names)

try:
    plotting_coupling_matrix(d2_weights,d1_weights)
except NameError:
    plotting_coupling_matrix(d2_weights,d2_weights*0)


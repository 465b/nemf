#!/usr/bin/env python
# coding: utf-8

# # NPZD Model

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def euler_forward(d0,d1_weights,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    time = time_step * time_step_size
    #print(d1_weights)
    #print(np.matmul(d1_weights,d0)*time_step_size)
    d0 = d0 + np.matmul(d1_weights,d0)*time_step_size
    time_step += 1
    
    return d0, time_step

def run_time_evo(method,model,max_time,time_step_size,d0,constants):
    
    # calculate all time constants
    n_obs = len(d0)
    n_steps = int(max_time/time_step_size)
    
    # initialize data arrays
    d0_log = np.zeros( (n_steps,n_obs) )
    d0_log[0] = d0
    d1_weights = np.zeros( (4,4) )
    
    try:
        I
    except NameError:
        I = 0
    
    # calculate the time evolution
    time_step = 0
    for ii in np.arange(1,n_steps):
        d1_weights = model(d0_log[ii-1,0],d0_log[ii-1,1],
                           d0_log[ii-1,2],I,d1_weights,constants)
        d0_log[ii],time_step = method(d0_log[ii-1],d1_weights,
                                      time_step,time_step_size)
        
    return d0_log, time_step_size

""" initial values and coupling weights are taken
    from doi.org/10.1016/j.ecolmodel.2013.01.012 """ 


def LLM_model(N,P,Z,I,d1_weights,constants):
    [k_N,k_I,mu_m,epsilon,g,phi_Z,phi_Z_star,phi_P,phi_P_star,gamma_m,beta] = constants
    
    J_val = J(N,I,k_N,k_I,mu_m)
    G_val = G(epsilon,g,P)

    d1_weights[0,1] = -J_val
    d1_weights[0,2] = phi_Z
    d1_weights[0,3] = gamma_m

    d1_weights[1,1] = J_val - phi_P
    d1_weights[1,2] = -G_val

    d1_weights[2,2] = beta*G_val - phi_Z

    d1_weights[3,1] = phi_P
    d1_weights[3,2] = (1-beta)*G_val
    d1_weights[3,3] = -gamma_m
    
    return d1_weights


def LQM_model(N,P,Z,I,d1_weights,constants):
    [k_N,k_I,mu_m,epsilon,g,phi_Z,phi_Z_star,phi_P,phi_P_star,gamma_m,beta] = constants
    
    J_val = J(N,I,k_N,k_I,mu_m)
    G_val = G(epsilon,g,P)

    d1_weights[0,1] = -J_val
    d1_weights[0,2] = phi_Z
    d1_weights[0,3] = gamma_m

    d1_weights[1,1] = J_val - phi_P
    d1_weights[1,2] = -G_val

    d1_weights[2,2] = beta*G_val - phi_Z - phi_Z_star*Z

    d1_weights[3,1] = phi_P
    d1_weights[3,2] = (1-beta)*G_val + phi_Z_star*Z
    d1_weights[3,3] = -gamma_m
    
    return d1_weights

def QQM_model(N,P,Z,I,d1_weights,constants):
    [k_N,k_I,mu_m,epsilon,g,phi_Z,phi_Z_star,phi_P,phi_P_star,gamma_m,beta] = constants
    
    J_val = J(N,I,k_N,k_I,mu_m)
    G_val = G(epsilon,g,P)

    d1_weights[0,1] = -J_val + phi_P
    d1_weights[0,2] = phi_Z
    d1_weights[0,3] = gamma_m

    d1_weights[1,1] = J_val - phi_P - phi_P_star*P
    d1_weights[1,2] = -G_val

    d1_weights[2,2] = beta*G_val - phi_Z - phi_Z_star*Z

    d1_weights[3,1] = phi_P*P
    d1_weights[3,2] = (1-beta)*G_val + phi_Z_star*Z
    d1_weights[3,3] = -gamma_m
    
    return d1_weights



# coupling functions
def J(N,I,k_N,k_I,mu_m):
    """ we are currenly assuming constant, perfect and homogenious illumination.
        hence, the f_I factor is currently set to 1 """
    f_N = N/(k_N+N)
    f_I = 1 #I/(k_I+I)
    J_val = mu_m*f_N*f_I
    return J_val



def G(epsilon,g,P):
    """ Holling type III function """
    G_val = (g*epsilon*P**2)/(g+(epsilon*P**2))
    return G_val


def heinle_2013():

    C_ref = 1.066  # currently unused, would be in I()
    d = 1  # currently unused, would be in I()

    beta = 0.925

    mu_m = 0.270 
    #u_m = 0.50 
    epsilon = 1.6
    gamma_m = 0.048
    g = 1.575

    phi_P = 0.040
    phi_P_star = 0.025
    phi_P_star = 0
    phi_Z = 0.010
    phi_Z_star = 0.340
    phi_Z_star = 0

    k_N = 0.700
        

    """ we are currenly assuming constant, perfect and homogenious illumination.
        hence, the f_I factor is currently set to 1 and I has no influence """
    alpha = 0.256 # currently unused, would be in I()
    k_I = 0.07 # find source for this
    I = 1 # find source for this!

    constants = [k_N,k_I,mu_m,epsilon,g,
                 phi_Z,phi_Z_star,phi_P,phi_P_star,
                 gamma_m,beta]
    
    return constants
    
'''
def I(C_ref,c,alpha):
    """ See DOI: 10.1357/002224003322981147
        if interested to parametrize I """
    pass
'''    



# In[ ]:


names =      ['N','P','Z','D']
d0 = np.array([1,  1,  1,  1])

time_step_size = 1/356
max_time = 100

method = euler_forward
model = LLM_model

constants = heinle_2013()

d0_log,time_step_size = run_time_evo(method,model,
                                     max_time,time_step_size,
                                     d0,constants)

plt.figure(figsize=(8,6))
time = time_step_size*np.arange(np.shape(d0_log)[0])
for ii in np.arange(np.shape(d0_log)[1]):
    plt.plot(time,d0_log[:,ii],label=names[ii])

plt.title('NPZD model behaviour over time')
plt.ylabel('Concentration [mmol N m−3]')
plt.xlabel('Time [y]')
plt.legend()
plt.show()


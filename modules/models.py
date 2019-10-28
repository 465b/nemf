import numpy as np
from . import caller


# Time Evolution

## Integration Schemes
def euler_forward(ODE_state,ODE_coeff,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    #print(ODE_state,ODE_coeff)
    ODE_state = ODE_state + np.matmul(ODE_coeff,ODE_state)*time_step_size
    
    return ODE_state

def runge_kutta(ODE_state,ODE_coeff,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    ODE_state_half = ODE_state + time_step_size/2*np.matmul(ODE_coeff,ODE_state)
    ODE_state = ODE_state_half + np.matmul(ODE_coeff,ODE_state_half)*time_step_size
    
    return ODE_state


## ODE coefficient models
""" all weights model have the form: model(ODE_coeff).
    no ODE_state dependence (so far neccessary) and all required
    constants should be called in the model through a function 
    (i dont know if thats very elegant, but saves an uneccessary )"""


def standard_weights_model(ODE_state,ODE_coeff):
    return ODE_coeff


## Fit models
def direct_fit_model(integration_scheme, time_evo_max, dt_time_evo, ODE_state, ODE_coeff=None, 
                       ODE_coeff_model=standard_weights_model,
                       stability_rel_tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
    """ NPZD model """
    
    F_ij,is_stable = caller.run_time_evo(integration_scheme, time_evo_max,dt_time_evo,ODE_state,ODE_coeff_model,ODE_coeff,
                stability_rel_tolerance,tail_length_stability_check,start_stability_check)
    F_i = F_ij[-1]

    return F_i,is_stable


def standard_fit_model(integration_scheme, time_evo_max, dt_time_evo, ODE_state, ODE_coeff=None, 
                       ODE_coeff_model=standard_weights_model,
                       stability_rel_tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):

    F_ij, is_stable = caller.run_time_evo(integration_scheme, time_evo_max,dt_time_evo,ODE_state,ODE_coeff_model,ODE_coeff,
                stability_rel_tolerance,tail_length_stability_check,start_stability_check)
    F_i = F_ij[-1]
    prediction = np.array(np.sum(F_i) - 2*F_i[-1])

    return prediction, is_stable


# Gradiant Decent

## Gradient Decent Methods

def SGD_basic(free_param,gradient,grad_scale):
    """ construct the gradient of the cost function  
    to minimize it with an 'SGD' approach """
    
    free_param_next = free_param[-1] - grad_scale*gradient
    
    return free_param_next


def SGD_momentum(free_param,gradient,grad_scale):
    """ construct the gradient of the cost function  
        to minimize it with an 'SGD-momentum' approach """
    
    if len(free_param) >= 3:
        previous_delta = free_param[-2]-free_param[-3]
    else:
        previous_delta = 0

    alpha = 1e-1 # fix function to accept alpha as an input
    free_param_next = free_param[-1] - grad_scale*gradient + alpha*previous_delta
    
    return free_param_next


# NPZD model functions

## Grazing Models

def J(N,k_N,mu_m):
    """ Nutrition saturation model"""

    """ we are currenly assuming constant, perfect 
        and homogenious illumination. Hence, the 
        f_I factor is currently set to 1 """

    f_N = N/(k_N+N)
    f_I = 1 #I/(k_I+I)
    cost_val = mu_m*f_N*f_I
    return cost_val


def Grazing_typeII(g,k_P,P):
    """ Holling type II function """
    G_val = g*(P/(k_P + P))
    return G_val


def Grazomg_typeIII(epsilon,g,P):
    """ Holling type III function """
    G_val = (g*epsilon*P**2)/(g+(epsilon*P**2))
    return G_val


# NPZD models

def LLM_model(ODE_state,ODE_coeff):
    """ See doi.org/10.1016/j.ecolmodel.2013.01.012 """

    constants = heinle_2013()
    
    [mort_P,mort_Z,mort_F,
     mort_P_square,mort_Z_square,mort_F_square,
     exc_P,exc_Z,exc_F,    
     beta_Z,beta_F,
     grazmax_Z,grazmax_F,
     grazenc_Z, grazenc_F,
     k_I,k_N,k_P,k_Z,
     mu_max,gamma] = constants
    
    N,P,Z = ODE_state[0],ODE_state[1],ODE_state[2]

    cost_val = J(N,k_N,mu_max)
    G_val = Grazomg_typeIII(grazenc_Z,grazmax_Z,P)

    ODE_coeff = np.zeros((4,4))

    ODE_coeff[0,1] = -cost_val
    ODE_coeff[0,2] = mort_Z
    ODE_coeff[0,3] = gamma

    ODE_coeff[1,1] = cost_val - mort_P
    ODE_coeff[1,2] = -G_val

    ODE_coeff[2,2] = beta_Z*G_val - mort_Z

    ODE_coeff[3,1] = mort_P
    ODE_coeff[3,2] = (1-beta_Z)*G_val
    ODE_coeff[3,3] = -gamma
    
    return ODE_coeff


def LQM_model(ODE_state,ODE_coeff):
    """ See doi.org/10.1016/j.ecolmodel.2013.01.012 """

    constants = heinle_2013()
    
    [mort_P,mort_Z,mort_F,
     mort_P_square,mort_Z_square,mort_F_square,
     exc_P,exc_Z,exc_F,    
     beta_Z,beta_F,
     grazmax_Z,grazmax_F,
     grazenc_Z, grazenc_F,
     k_I,k_N,k_P,k_Z,
     mu_max,gamma] = constants
    
    N,P,Z = ODE_state[0],ODE_state[1],ODE_state[2]

    cost_val = J(N,k_N,mu_max)
    G_val = Grazomg_typeIII(grazenc_Z,grazmax_Z,P)

    ODE_coeff = np.zeros((4,4))

    ODE_coeff[0,1] = -cost_val
    ODE_coeff[0,2] = mort_Z
    ODE_coeff[0,3] = gamma

    ODE_coeff[1,1] = cost_val - mort_P
    ODE_coeff[1,2] = -G_val

    ODE_coeff[2,2] = beta_Z*G_val - mort_Z - mort_Z_square*Z

    ODE_coeff[3,1] = mort_P
    ODE_coeff[3,2] = (1-beta_Z)*G_val + mort_Z_square*Z
    ODE_coeff[3,3] = -gamma
    
    return ODE_coeff


def QQM_model(ODE_state,ODE_coeff):
    """ See doi.org/10.1016/j.ecolmodel.2013.01.012 """

    constants = heinle_2013()
    
    [mort_P,mort_Z,mort_F,
     mort_P_square,mort_Z_square,mort_F_square,
     exc_P,exc_Z,exc_F,    
     beta_Z,beta_F,
     grazmax_Z,grazmax_F,
     grazenc_Z, grazenc_F,
     k_I,k_N,k_P,k_Z,
     mu_max,gamma] = constants

    N,P,Z = ODE_state[0],ODE_state[1],ODE_state[2]

    ODE_coeff = np.zeros((4,4))
    
    cost_val = J(N,k_N,mu_max)
    G_val = Grazomg_typeIII(grazenc_Z,grazmax_Z,P)

    ODE_coeff[0,1] = -cost_val + mort_P
    ODE_coeff[0,2] = mort_Z
    ODE_coeff[0,3] = gamma

    ODE_coeff[1,1] = cost_val - mort_P - mort_P_square*P
    ODE_coeff[1,2] = -G_val

    ODE_coeff[2,2] = beta_Z*G_val - mort_Z - mort_Z_square*Z

    ODE_coeff[3,1] = mort_P*P
    ODE_coeff[3,2] = (1-beta_Z)*G_val + mort_Z_square*Z
    ODE_coeff[3,3] = -gamma
    
    return ODE_coeff


# NPZFD models
def NPZFD_model(ODE_state,ODE_coeff):
    """ basic extended NPZD model with a set of stable
        constants (found by hand by a student) """

    constants = NPZFD_constants()
    
    [mort_P,mort_Z,mort_F,
     mort_P_square,mort_Z_square,mort_F_square,
     exc_P,exc_Z,exc_F,    
     beta_Z,beta_F,
     grazmax_Z,grazmax_F,
     grazenc_Z, grazenc_F,
     k_I,k_N,k_P,k_Z,
     mu_max,gamma] = constants
    
    N,P,Z = ODE_state[0],ODE_state[1],ODE_state[2]
        
    cost_val = J(N,k_N,mu_max)
    G_val_Z = Grazing_typeII(grazmax_Z,k_P,P)
    G_val_F= Grazing_typeII(grazmax_F,k_Z,Z)

    ODE_coeff = np.zeros((4,4))
    
    # N
    ODE_coeff[0,1] = -cost_val + exc_P
    ODE_coeff[0,2] = exc_Z
    ODE_coeff[0,3] = exc_F
    ODE_coeff[0,4] = gamma
    
    # P
    ODE_coeff[1,1] = cost_val - exc_P - mort_P
    ODE_coeff[1,2] = - G_val_Z

    # Z
    ODE_coeff[2,2] = beta_Z*G_val_Z - mort_Z - exc_Z
    ODE_coeff[2,3] = -G_val_F
    
    # prediction
    ODE_coeff[3,3] = beta_F*G_val_F - mort_F - exc_F

    ODE_coeff[4,1] = mort_P
    ODE_coeff[4,2] = mort_Z + (1-beta_Z)*G_val_Z
    ODE_coeff[4,3] = mort_F + (1-beta_F)*G_val_F
    ODE_coeff[4,4] = -gamma
    
    return ODE_coeff


def NPZFD_constants():
    # standard time measure is one day
    
    # mortality rate per day
    mort_P = 0.05
    mort_Z = 0.1
    mort_F = 0.142
    
    mort_P_square = 0
    mort_Z_square = 0
    mort_F_square = 0
    
    # excretion rate per day
    exc_P = 0.000001
    exc_Z = 0.000002
    exc_F = 0.00001
    
    # consumtion/assimilation efficiency
    beta_Z = 0.95
    beta_F = 0.69
    
    # maximal grazing rate per day
    grazmax_Z = 1
    grazmax_F = 0.3
    
    # grazing encounter rate (unused)
    grazenc_Z = 0 
    grazenc_F = 0
        
    # half efficency points [mmolNm**-3] (assuming x*(k+x) growth)
    k_I = 0 # (unused)
    k_N = 0.5
    k_P = 0.5
    k_Z = 0.19
    
    # planktion max growth rate per day
    mu_max = 2
    
    # remineralisation rate per day
    gamma = 0.08
    
    constants = [mort_P,mort_Z,mort_F,
                 mort_P_square,mort_Z_square,mort_F_square,
                 exc_P,exc_Z,exc_F,    
                 beta_Z,beta_F,
                 grazmax_Z,grazmax_F,
                 grazenc_Z, grazenc_F,
                 k_I,k_N,k_P,k_Z,
                 mu_max,gamma]
    
    return constants


def heinle_2013():
    """ See doi.org/10.1016/j.ecolmodel.2013.01.012 """

        # standard time measure is one day
    
    # mortality rate per day
    mort_P = 0.04
    mort_Z = 0.01
    mort_F = 0
    
    mort_P_square = 0.025
    mort_Z_square = 0.34
    mort_F_square = 0
    
    # excretion rate per day
    exc_P = 0.000001
    exc_Z = 0.000002
    exc_F = 0
    
    # consumtion/assimilation efficiency
    beta_Z = 0.925
    beta_F = 0
    
    # maximal grazing rate per day
    grazmax_Z = 1.575
    grazmax_F = 0.3
    
    # grazing encounter rate (unused)
    grazenc_Z = 1.6 
    grazenc_F = 0
        
    # half efficency points [mmolNm**-3] (assuming x*(k+x) growth)
    k_I = 0 # (unused)
    k_N = 0.7
    k_P = 0
    k_Z = 0
    
    # planktion max growth rate per day
    mu_max = 0.270
    
    # remineralisation rate per day
    gamma = 0.048
    
    constants = [mort_P,mort_Z,mort_F,
                 mort_P_square,mort_Z_square,mort_F_square,
                 exc_P,exc_Z,exc_F,    
                 beta_Z,beta_F,
                 grazmax_Z,grazmax_F,
                 grazenc_Z, grazenc_F,
                 k_I,k_N,k_P,k_Z,
                 mu_max,gamma]
    
    """ we are currenly assuming constant, perfect and homogenious illumination.
        hence, the f_I factor is currently set to 1 and I has no influence """
    alpha = 0.256 # currently unused, would be in I()
    I = 1 # find source for this!

    return constants





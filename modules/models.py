import numpy as np
from . import caller


# Time Evolution

## Integration Schemes
def euler_forward(ODE_state,ODE_coeff,dt_time_evo):
    """ integration scheme for the time evolution 
        based on a euler forward method
        
    Parameters:
    -----------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    
    Returns:
    --------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE. Now at the next iteration step  """
    
    ODE_state = ODE_state + np.matmul(ODE_coeff,ODE_state)*dt_time_evo 
    
    return ODE_state

def runge_kutta(ODE_state,ODE_coeff,dt_time_evo):
    """ integration scheme for the time evolution 
        based on a euler forward method 
        
    Parameters:
    -----------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    
    Returns:
    --------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE. Now at the next iteration step  """
    
    ODE_state_half = ODE_state + dt_time_evo/2*np.matmul(ODE_coeff,ODE_state)
    ODE_state = ODE_state_half + np.matmul(ODE_coeff,ODE_state_half)*dt_time_evo
    
    return ODE_state


## ODE coefficient models
""" all weights model have the form: model(ODE_state,ODE_coeff).
    no ODE_state dependence (so far necessary) and all required
    constants should be called in the model through a function 
    (i don't know if thats very elegant, but saves an unnecessary)"""


def standard_weights_model(ODE_state,ODE_coeff):
    """ Models if no implicit time dependency is present.
        Hence, ODE_coeff is constant and gets returned unchanged.
        
    Parameters:
    -----------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
        
    Returns:
    --------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE  """

    return ODE_coeff


## Fit models
""" Fit models define how the output of the time evolution (if stable) is used
    for further processing. Then, the optimization routine uses this processed
    time evolution output to fit the model to. Hence, the name fit_model. """
    
def direct_fit_model(integration_scheme, time_evo_max, dt_time_evo,
                     ODE_state, ODE_coeff=None, 
                     ODE_coeff_model=standard_weights_model,
                     stability_rel_tolerance=1e-5,tail_length_stability_check=10,
                     start_stability_check=100):
    """ Returns the last step in the time evolution. Hence, it uses the values
        of the time evolution *directly*, without any further processing 
    
    Parameters:
    -----------
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    time_evo_max
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    ODE_state : numpy.array
        1D array containing the initial state of the oberserved quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative flucuation range in the tail
        of the time evolutoin. If below, system is called stable.
    tail_length_stability_check : postive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned

    Returns:
    --------
    F_i : numpy.array
        2D-array containing the output of the time evolution
        after the fit_model has been applied, stacked along the first axis.
    is_stable : bool
        true if stability conditions are met. 
        See verify_stability_time_evolution() for more details. """
    
    F_ij, is_stable = caller.run_time_evo(integration_scheme, time_evo_max,
                                          dt_time_evo,ODE_state,
                                          ODE_coeff_model,ODE_coeff,
                                          stability_rel_tolerance,
                                          tail_length_stability_check,
                                          start_stability_check)
    F_i = F_ij[-1]

    return F_i,is_stable


def net_flux_fit_model(integration_scheme, time_evo_max, dt_time_evo,
                       idx_source, idx_sink,
                       ODE_state, ODE_coeff=None,
                       ODE_coeff_model=standard_weights_model,
                       stability_rel_tolerance=1e-5,
                       tail_length_stability_check=10,
                       start_stability_check=100):

    """ Takes the last step in the time evolution and calculates the sum
        of its entries. Counts the last entry negatively.
        This is done to represent a 'dump' through which the positive
        net-flux of the system is compensated.
        
    Parameters:
    -----------
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    time_evo_max : float
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo : float
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    b
    ODE_state : numpy.array
        1D array containing the initial state of the oberserved quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative flucuation range in the tail
        of the time evolutoin. If below, system is called stable.
    tail_length_stability_check : postive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned

    Returns:
    --------
    F_i : numpy.array
        2D-array containing the output of the time evolution
        after the fit_model has been applied, stacked along the first axis.
    is_stable : bool
        true if stability conditions are met. 
        See verify_stability_time_evolution() for more details. """
    
    F_ij, is_stable = caller.run_time_evo(integration_scheme, time_evo_max,
                                          dt_time_evo,ODE_state,
                                          ODE_coeff_model,ODE_coeff,
                                          stability_rel_tolerance,
                                          tail_length_stability_check,
                                          start_stability_check)
    F_i = F_ij[-1]
    prediction = np.array(np.sum(F_i[idx_source]) - np.sum(F_i[idx_sink]))

    return prediction, is_stable


# Gradient Decent

## Gradient Decent Methods

def SGD_basic(free_param,gradient,grad_scale):
    """ Stochastic Gradient Descent (SGD) implementation to minimize
        the cost function 
        
    Parameters:
    -----------
    free_param : numpy.array
        2D-array containing the set of optimized free parameter,
        stacked along the first axis.
    gradient : numpy.array
        Gradient at the center point calculated by the randomly chosen 
        local environment. The gradient always points in the direction
        of steepest ascent.
    grad_scale : positive float
        Scales the step size in the gradient descent. Often also
        referred to as learning rate. Necessary to compensate for the
        "roughness" of the objective function field.

    Returns:
    -------
    free_param_next : numpy.array
        1D-array containing the set of optimized free parameter
        for the next iteration step. """
    
    
    free_param_next = free_param[-1] - grad_scale*gradient
    
    return free_param_next


def SGD_momentum(free_param,gradient,grad_scale):
    """ Stochastic Gradient Descent (SGD) implementation plus and additional
        momentum term to minimize the cost function.
    
    Parameters:
    -----------
    free_param : numpy.array
        2D-array containing the set of optimized free parameter,
        stacked along the first axis.
    gradient : numpy.array
        Gradient at the center point calculated by the randomly chosen 
        local environment. The gradient always points in the direction
        of steepest ascent.
    grad_scale : positive float
        Scales the step size in the gradient descent. Often also
        referred to as learning rate. Necessary to compensate for the
        "roughness" of the objective function field.

    Returns:
    -------
    free_param_next : numpy.array
        1D-array containing the set of optimized free parameter
        for the next iteration step. """
    
    if len(free_param) >= 3:
        previous_delta = free_param[-2]-free_param[-3]
    else:
        previous_delta = 0

    alpha = 1e-1 # fix function to accept alpha as an input
    free_param_next = free_param[-1] - grad_scale*gradient + alpha*previous_delta
    
    return free_param_next

# Baltic Compartment Coefficient models

## 5-compartment models / 5-dimensional model
def d5_model(ODE_state, constants):

    ppr, pco, fis, bac, det, out = ODE_state
    f_pco_ppr, h_pco_ppr = 1,1
    f_pco_bac, h_pco_bac = 1,1
    f_fis_pco, h_fis_pco = 1,1
    f_bac_det, h_bac_det = 1,1

    
    mort_ppr = 0
    mort_pco = 0.01
    mort_fis = 0.01
    mort_bac = 0.01
    mort_fis = 0.01
    mort_det = 0.10

    ODE_coeff = np.zeros((6,6))

    # PPR
    # 0 because PPR is a artificial source

    # PCO
    ODE_coeff[1,1] = holling_typeII(f_pco_ppr,h_pco_ppr,ppr) + holling_typeII(f_pco_bac,h_pco_bac,bac) - holling_type0(mort_pco)
    ODE_coeff[1,2] = - holling_typeII(f_fis_pco,h_fis_pco,pco)

    # FIS
    ODE_coeff[2,2] = holling_typeII(f_fis_pco,h_fis_pco,pco) - holling_type0(mort_fis)
    
    # BAC
    ODE_coeff[3,1] = - holling_typeII(f_pco_bac,h_pco_bac,bac)
    ODE_coeff[3,3] = holling_typeII(f_bac_det,h_bac_det,det) - holling_type0(mort_bac)

    # DET
    ODE_coeff[4,0] = holling_type0(mort_ppr)
    ODE_coeff[4,1] = holling_type0(mort_pco)
    ODE_coeff[4,2] = holling_type0(mort_fis)
    ODE_coeff[4,3] = holling_type0(mort_bac) - holling_typeII(f_bac_det,h_bac_det,det)
    ODE_coeff[4,4] = - holling_type0(mort_det)

    # SINK
    ODE_coeff[5,5] = holling_type0(mort_det)

    return ODE_coeff 

    
# NPZD model functions

## Grazing Models

def J(N,k_N,mu_m):
    """ Nutrition saturation model"""

    """ we are currently assuming constant, perfect 
        and homogeneous illumination. Hence, the 
        f_I factor is currently set to 1 """

    f_N = N/(k_N+N)
    f_I = 1 #I/(k_I+I)
    cost_val = mu_m*f_N*f_I
    return cost_val

def holling_type0(value):
    return value

def holling_typeII(food_processing_time,hunting_rate,prey_population):
    """ Holling type II function """
    consumption_rate = ((hunting_rate * prey_population)/
            (1+hunting_rate * food_processing_time * prey_population))
    return G_val


def holling_typeIII(epsilon,g,P):
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
    G_val_Z = holling_typeII(grazmax_Z,k_P,P)
    G_val_F= holling_typeII(grazmax_F,k_Z,Z)

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





# Example D5 Model

import modules.caller
import modules.models
import modules.plot
import numpy as np

path_ODE_state_init = 'initial_values/ODE_state_D05_init.tsv'
path_ODE_coeff_init = "initial_values/ODE_coeff_D05_init.tsv"
y = np.array([0])

sample_sets = 3

free_param_stack, prediction_stack, cost_stack = modules.caller.dn_monte_carlo(
                    path_ODE_state_init,path_ODE_coeff_init,y,
                    fit_model = modules.models.net_flux_fit_model,
                    gradient_method = modules.models.SGD_basic,
                    integration_method = modules.models.euler_forward,
                    ODE_coeff_model = modules.models.standard_weights_model,
                    barrier_slope=1e-6,
                    sample_sets = sample_sets,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    stability_rel_tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=50,
                    seed=137)

for ii in np.arange(sample_sets):
    modules.plot.XFL(free_param_stack[ii],prediction_stack[ii])


# Example NPZD Model

"""
import modules.caller
import modules.models
import modules.plot
import numpy as np

path_ODE_state_init = 'initial_values/d0_NPZD_heinle.tsv'
path_ODE_coeff_init = 'initial_values/ODE_coeff_D05_init.tsv'
y = np.array([1,1,1,1])

sample_sets = 3

free_param,prediction = modules.caller.NPZD_monte_carlo(
                    path_ODE_state_init,path_ODE_coeff_init,y,
                    fit_model = modules.models.direct_fit_model,
                    gradient_method = modules.models.SGD_basic,
                    integration_method = modules.models.euler_forward,
                    ODE_coeff_model = modules.models.LLM_model,
                    barrier_slope=1e-6,
                    sample_sets = 3,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    stability_rel_tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137)
                    
for ii in np.arange(sample_sets):
    modules.plot.XFL(free_param[ii],prediction[ii])
"""
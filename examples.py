# Example D5 Model

import modules.caller
import modules.models
import modules.plot
import numpy as np

path_d0_init = 'initial_values/d1_D05_init.tsv'
path_d1_init = "initial_values/d2_D05_init.tsv"
y = np.array([0])

X, F = modules.caller.dn_monte_carlo(path_d0_init,path_d1_init,y,
                fit_model = modules.models.standard_fit_model,
                gradient_method = modules.models.SGD_basic,
                integration_method = modules.models.euler_forward,
                d1_weights_model = modules.models.standard_weights_model,
                mu=1e-6,
                sample_sets = 3,
                gd_max_iter=300,
                time_evo_max=300,
                dt_time_evo=1/5,
                pert_scale=1e-4,
                grad_scale=1e-12,
                tolerance=1e-5,
                tail_length_stability_check=10,
                start_stability_check=100,
                seed=137)


modules.plot.XFL(X[0],F[0])


# Example NPZD Model

"""
import modules.caller
import modules.models
import modules.plot
import numpy as np

path_d0_init = 'initial_values/d0_NPZD_heinle.tsv'
path_d1_init = 'initial_values/d2_D05_init.tsv'
y = np.array([1,1,1,1])

X,F = modules.caller.NPZD_monte_carlo(path_d0_init,path_d1_init,y,
                    fit_model = modules.models.direct_fit_model,
                    gradient_method = modules.models.SGD_basic,
                    integration_method = modules.models.euler_forward,
                    d1_weights_model = modules.models.LLM_model,
                    mu=1e-6,
                    sample_sets = 3,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137)
                    
modules.plot.XFL(X[0],F[0])
"""
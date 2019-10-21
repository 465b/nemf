import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# plotting routines

def coupling_matrix(d2_weights,d1_weights,names):
    plt.figure(figsize=(12,6))
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
    plt.savefig('coupling_matrices.svg')
    plt.show()


def time_evolution(d0_log,d1_log,time_step_size,names):
    """ plotting_time_evolution(d0_log=d0_log,d1_log=d1_log,
        time_step_size=time_step_size,names=names) """
    # plotting
    time = np.arange(np.shape(d0_log)[0]) * time_step_size 

    plt.figure(figsize=(10,3))
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
    plt.savefig('individual_flux_mass.svg')
    plt.show()

    
    plt .figure(figsize=(10,3))
    
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
    plt.savefig('total_flux_mass.svg')
    plt.show()


def XFL(X=None,F=None,L=None,context='talk'):
    
    sns.set_context(context)
    
    if X is not None:
        plt.figure()
        plt.title('Free Input Parameter')
        plt.plot(X[:-1])
        plt.ylabel('Input value (arb. u.)')
        plt.ylabel('Iteration Step')
        plt.show()

    if F is not None:
        plt.figure()
        plt.title('Time Evolution Output')
        plt.plot(F[:-1])
        plt.ylabel('Output value (arb. u.)')
        plt.xlabel('Iteration Step')
        plt.show()

    if L is not None:
        plt.figure()
        plt.title('Loss function over time')
        plt.plot(L[:-1])
        plt.ylabel('Loss function')
        plt.xlabel('Iteration Step')
        plt.show()
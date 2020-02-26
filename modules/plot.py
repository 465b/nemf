import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

sns.set_style('whitegrid')
sns.set_context('paper')

# plotting routines

def interaction_graph(model_config):
    """ Takes the model configuration and draws a labeled
        directional-multi-graph to illustrate the interactions """
    
    # define the present nodes/compartments
    nodes = list(model_config.states)

    # fetch the list of interaction paths present
    interactions = list(model_config.interactions)

    # fetch list of edges and their labels
    edges = []; labels = []
    for path in interactions:
        for edge in model_config.interactions[path]:
            # swap direction depending on the function sign 
            if edge['sign'] == '+1':
                edges.append(tuple(path.split(':'))[::-1])
            else:
                edges.append(tuple(path.split(':')))
            labels.append(edge['fkt'])

    # setting up edge labes dict
    edge_labels = {}
    for ii,edge in enumerate(edges):
        # checks if key is already present
        if (edge in edge_labels):
            # and if so appends it
            edge_labels[edge] += '\n + {}'.format(labels[ii].replace('_','\n'))
        else:
            # or creates it
            edge_labels[edge] = labels[ii].replace('_','\n')

    # initialise graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # define node positions
    pos = nx.kamada_kawai_layout(G)

    # actual plotting
    nx.draw(G, pos,node_size=2000, node_color='pink',
           labels={node:node for node in G.nodes()},
           arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
        label_pos=0.35, font_size=10,font_color='tab:red',rotate=False)
    plt.show()


def coupling_matrix(d2_weights,ODE_coeff_weights,names):
    plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title("d2 coupling matrix")
    plt.imshow(d2_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()

    ax = plt.subplot(122)
    ax.set_title("ODE_coeff coupling matrix")
    plt.imshow(ODE_coeff_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()
    plt.savefig('coupling_matrices.svg')
    plt.show()


def time_evolution(ODE_state_log,ODE_coeff_log,time_step_size,names):
    """ plotting_time_evolution(ODE_state_log=ODE_state_log,ODE_coeff_log=ODE_coeff_log,
        time_step_size=time_step_size,names=names) """
    # plotting
    time = np.arange(np.shape(ODE_state_log)[0]) * time_step_size 

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    for kk in np.arange(len(ODE_coeff_log[0])):
        plt.title( "Absolute Carbon Mass")
        plt.plot(time,ODE_state_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Absolute Carbon mass [mg]")
        plt.xlabel("Time [years]")
    

    plt.subplot(122)
    for kk in np.arange(len(ODE_coeff_log[0])):
        plt.title( "Carbon Mass Flux")
        plt.plot(time,ODE_coeff_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Carbon mass change [mg per yer]")
        plt.xlabel("Time [years]")
    plt.tight_layout()
    plt.savefig('individual_flux_mass.svg')
    plt.show()

    
    plt .figure(figsize=(10,3))
    
    plt.subplot(121)
    plt.plot(time, np.sum(ODE_state_log,axis = 1) - 2*ODE_state_log[:,-1])
    plt.title("Total Carbon mass over time")
    plt.ylabel("Total Carbon mass [mg]")
    plt.xlabel('Time [years]')

    plt.subplot(122)
    plt.plot(time, np.sum(ODE_coeff_log,axis = 1) - 2*ODE_coeff_log[:,-1] )
    plt.title("Total Carbon mass flux over time")
    plt.ylabel("Total Carbon mass flux [mg/y]")
    plt.xlabel('Time [years]')
    plt.tight_layout()
    plt.savefig('total_flux_mass.svg')
    plt.show()


def XFL(free_param=None,prediction=None,cost=None,context='talk'):
    
    sns.set_context(context)
    
    if free_param is not None:
        plt.figure()
        plt.title('Free Input Parameter')
        plt.plot(free_param[:-1])
        plt.ylabel('Input value (arb. u.)')
        plt.ylabel('Iteration Step')
        plt.show()

    if prediction is not None:
        plt.figure()
        plt.title('Time Evolution Output')
        plt.plot(prediction[:-1])
        plt.ylabel('Output value (arb. u.)')
        plt.xlabel('Iteration Step')
        plt.show()

    if cost is not None:
        plt.figure()
        plt.title('Loss function over time')
        plt.plot(cost)
        plt.ylabel('Loss function')
        plt.xlabel('Iteration Step')
        plt.show()
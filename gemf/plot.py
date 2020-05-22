import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import yaml
import gemf
from copy import deepcopy

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette("husl")

# plotting routines


def draw_interaction_graph(model):
    """ Takes the model configuration and draws a labeled
        directional-multi-graph to illustrate the interactions """
    
    nodes = list(model.compartment)
    interactions = list(model.interactions)

    # --- CONFIGURATION DUMP ---
    config = model.configuration.copy()
    if 'idx_sinks' in config:
        config.pop('idx_sinks')
        config.pop('idx_sources')

    # turns dict into yaml style string
    ## quick and dirty reformating of the lists and tuples
    for item in config:
        if ((type(config[item]) == list) or (type(config[item]) == tuple)):
            config[item] = str(config[item])
    
    comment = yaml.dump(config, default_flow_style=False, line_break=True)
    comment = comment.replace('!!python/tuple','')

    # --- GRAPH ---
    # fetch list of edges and their labels
    edges = []; labels = []
    for path in interactions:
        for edge in model.interactions[path]:
            edges.append(tuple(path.split(':'))[::-1])
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
    pos = nx.circular_layout(G)

    # --- PLOTTING ---
    fig = plt.figure()
    ax = plt.subplot(111)
    # draws nodes
    nx.draw(G, pos,node_size=2000, node_color='pink',
           labels={node:node for node in G.nodes()},
           arrowsize=20)
    # draws edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
        label_pos=0.35, font_size=10,font_color='tab:red',rotate=False)

    # # adds configuration
    # plt.legend(title=comment,loc='center left', bbox_to_anchor=(1., 0.5))
    # 
    # # positions legends
    # ## Shrink current axis by 20%
    # box = ax.get_position()
    # ## Put a legend to the right of the current axis
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    return fig


def interaction_graph(model):
    """ shows a graph/network of all compartments and their interactions """
    fig = draw_interaction_graph(model)
    fig.tight_layout()
    plt.show()


def draw_cost(ax,cost):
    ax.title.set_text('cost')
    ax.set_ylabel('Cost function (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.plot(cost)

def draw_predictions(ax,predictions,model):
    if model.configuration['fit_model'] == 'direct_fit_model':
        labels = list(model.compartment)
    elif model.configuration['fit_model'] == 'net_flux_fit_model':
        labels = ['net incoming/outgoing flux']
    else:
        labels = []    
    ax.title.set_text('fit model prediction')
    handles = plt.plot(predictions)
    ax.set_ylabel('Model predictions (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.legend(handles, labels)


def draw_parameters(ax,parameters,model):
    labels = model.to_grad_method()[2]
    ax.title.set_text('parameters')
    handles = plt.plot(parameters)
    ax.set_ylabel('optimized parameters (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.legend(handles, labels, loc='upper left',bbox_to_anchor=(1.0, 1.0))
    

def draw_model_output(ax,model):
    time_series = model.log['time_series']
    t = time_series[:,0]
    y_t = time_series[:,1:]
    labels = list(model.compartment)

    handles = plt.plot(t,y_t)
    
    plt.legend(handles, labels)
    ax.set_ylabel('Model output (a.u.)')
    ax.set_xlabel('Time (a.u.)')
    plt.legend(handles, labels, loc='upper left',bbox_to_anchor=(1.0, 1.0))

    return ax


def draw_optimization_overview(model): 
    """ reads the data saved in the model class and depending on this data 
        chooses a visualization method to present the results """

    fig = plt.figure()
    fig.suptitle('Results of optimization run')
    
    if model.reference_data[:,0][0] == np.inf:
        non_steady_state = False
    else:
        non_steady_state = True
    
    if not np.isnan(model.log['cost'][0]): 

        ax1 = plt.subplot(221)
        draw_cost(ax1,model.log['cost'])
        ax2 = plt.subplot(222)
        draw_parameters(ax2,model.log['parameters'],model)
        ax3 = plt.subplot(212)

        if non_steady_state:
            t_ref =  model.reference_data[:,0]
            y_ref =  model.reference_data[:,1:]
            plt.plot(t_ref,y_ref,ls='--',linewidth=2)
            time_series = gemf.forward_model(model,t_eval=t_ref)
            time_series_model = time_series.log['time_series']
            ax2 = draw_model_output(ax2, model)
            ax2.title.set_text('optimized model')

        else: # steady-state
            t_max = model.configuration['max_time_evo']
            t_ref = np.linspace(0,t_max,1000)
            y_ref =  model.reference_data[:,1:]
            plt.hlines(y_ref,t_ref[0],t_ref[-1],ls='--')
            
            time_series_model = gemf.forward_model(model,t_eval=t_ref)
            ax3 = draw_model_output(ax3, time_series_model)
            ax3.title.set_text('optimized model')

    else:
        ax1 = plt.subplot(211)
        draw_parameters(ax1,model.log['parameters'],model)
        ax2 = plt.subplot(212)

        if non_steady_state:
            t_ref =  model.reference_data[:,0]
            y_ref =  model.reference_data[:,1:]
            plt.plot(t_ref,y_ref,ls='--',linewidth=2)
            time_series = gemf.forward_model(model,t_eval=t_ref)
            time_series_model = time_series.log['time_series']
            ax2 =  draw_model_output(ax2, model)
            ax2.title.set_text('optimized model')
        else: # steady-state
            t_max = model.configuration['max_time_evo']
            dt = model.configuration['dt_time_evo']
            t_ref = np.arange(0,t_max,dt)
            y_ref =  model.reference_data[:,1:]
            plt.hlines(y_ref,t_ref[0],t_ref[-1],ls='--')
            
            time_series_model = gemf.forward_model(model,t_eval=t_ref)
            draw_model_output(ax2, time_series_model)
            ax2.title.set_text('optimized model')

    return fig


def draw_output_summary(model):
    """ reads the data saved in the model class and depending on this data 
        chooses a visualization method to present the results with the help
        of draw_optimization_overview """
    
    if 'time_series' in model.log:
        # no optimization has happend.
        # hence, cost/predictions/parameters is 0-dim
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax = draw_model_output(ax,model)
        ax.title.set_text('Model Output')

    else:
        fig = draw_optimization_overview(model)

    return fig


def output_summary(model):
    """ reads the data saved in the model class and depending on this data 
        chooses a visualization method with the help of draw_output_summary
        to present the results """
    
    fig = draw_output_summary(model)
    plt.tight_layout()
    plt.show()


def initial_guess(model):
    ax = plt.subplot(111)

    t_ref =  model.reference_data[:,0]
    y_ref =  model.reference_data[:,1:]

    plt.title('Initial model behavior and reference')
    
    plt.plot(t_ref,y_ref,ls='--',linewidth=2)
    initial_model = deepcopy(gemf.forward_model(model,t_eval=t_ref))
    ax = draw_model_output(ax, initial_model)
    ax.title.set_text('Initial model guess')

    plt.show()

    return ax


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

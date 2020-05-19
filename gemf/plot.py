import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import yaml

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette("husl")

# plotting routines

def draw_interaction_graph(model_config):
    """ Takes the model configuration and draws a labeled
        directional-multi-graph to illustrate the interactions """
    
    nodes = list(model_config.compartment)
    interactions = list(model_config.interactions)

    # --- CONFIGURATION DUMP ---
    config = model_config.configuration.copy()
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

    # adds configuration
    plt.legend(title=comment,loc='center left', bbox_to_anchor=(1., 0.5))
    # positions legends
    ## Shrink current axis by 20%
    box = ax.get_position()
    ## Put a legend to the right of the current axis
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    return fig


def interaction_graph(model_config):
    fig = draw_interaction_graph(model_config)
    fig.tight_layout()
    plt.show()


def draw_cost(ax,cost):
    ax.title.set_text('cost')
    ax.set_ylabel('Cost function (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.plot(cost)

def draw_predictions(ax,predictions,model_config):
    if model_config.configuration['fit_model'] == 'direct_fit_model':
        labels = list(model_config.compartment)
    elif model_config.configuration['fit_model'] == 'net_flux_fit_model':
        labels = ['net incoming/outgoing flux']
    else:
        labels = []    
    ax.title.set_text('fit model prediction')
    handles = plt.plot(predictions)
    ax.set_ylabel('Model predictions (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.legend(handles, labels)


def draw_parameters(ax,parameters,model_config):
    labels = model_config.to_grad_method()[2]
    ax.title.set_text('parameters')
    handles = plt.plot(parameters)
    ax.set_ylabel('optimized parameters (a.u.)')
    ax.set_xlabel('Iteration Step')
    plt.legend(handles, labels)

def draw_model_output(ax,model,model_config):
    dt = model_config.configuration['dt_time_evo']
    T = model_config.configuration['time_evo_max']
    time = np.arange(T,step=dt)
    time = time[:len(model)]
    labels = list(model_config.compartment)

    fig = plt.figure()
    ax.title.set_text('optimized model')
    handles = plt.plot(time,model)
    plt.legend(handles, labels)
    ax.set_ylabel('Model output (a.u.)')
    ax.set_xlabel('Time (a.u.)')

    return fig


def draw_optimization_overview(cost,predictions,parameters,model
                            ,model_config,ii_sample=None): 

    if ii_sample is None:
        ii_sample = ''
    else:
        ii_sample = '#{}'.format(ii_sample)
    fig = plt.figure()
    fig.suptitle('Results of optimization run'+ii_sample)
    
    ax1 = plt.subplot(2,2,1)
    draw_cost(ax1,cost[:-1])
    ax2 = plt.subplot(2,2,2)
    draw_predictions(ax2,predictions[:-1],model_config)
    ax3 = plt.subplot(2,2,3)
    draw_parameters(ax3,parameters[:-1],model_config)
    ax4 = plt.subplot(2,2,4)
    draw_model_output(ax4, model,model_config)

    return fig


def draw_output_summary(model_config):
    log = model_config.log

    sample_sets_switch = len(np.shape(log['parameters']))
    if sample_sets_switch == 1:
        # no optimization has happend.
        # hence, cost/predictions/parameters is 0-dim
        print(log['cost'])
        print(log['predictions'])
        print(log['parameters'])
        ax = plt.subplot(1,1,1)
        fig = draw_model_output(ax,log['model'],model_config)
    elif sample_sets_switch == 2:
        fig = draw_optimization_overview(log['cost'],
                   log['predictions'],
                   log['parameters'],
                   log['model'][0],
                   model_config)
    elif sample_sets_switch == 3:
        for ii in np.arange(np.shape(log['parameters'])[0]):
            fig = draw_optimization_overview(log['cost'][ii],
                    log['predictions'][ii],
                    log['parameters'][ii],
                    log['model'][ii][np.where(np.isnan(log['cost'][ii]))[0][0]-1],
                    model_config)
    
    return fig

                   

def output_summary(model_config):
    fig = draw_output_summary(model_config)
    plt.tight_layout()
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

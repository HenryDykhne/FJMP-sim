import numpy, torch, dgl
import networkx as nx 
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def build_dag_graph(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()
    dag_graph.ndata["ground_truth_futures"] = graph.ndata["ground_truth_futures"].float()
    dag_graph.ndata["has_preds"] = graph.ndata["has_preds"].float()

    return dag_graph

def prune_graph_johnson(dag_graph, scene_idxs):
    """
    dag_graph: DGL graph with weighted edges
    graph contains edge property "edge_probs" which contains predicted probability of each edge type 

    Based on the predicted probabilities, prune graph until it is a DAG based on Johnson's algorithm

    Note that we can think of a batch of graphs as one big graph and apply the pruning procedure on the entire batch at once.
    """
    while True:
        # First identify cycles in graph
        G = dgl.to_networkx(dag_graph.cpu(), node_attrs=None, edge_attrs=None)
        cycles = nx.simple_cycles(G)

        eids = []
        count_cycles = 0
        
        start = time.time() #get rid of
        for cycle in cycles:
            out_cycle = torch.Tensor(cycle).to(dag_graph.device).long()
            in_cycle = torch.roll(out_cycle, 1)

            eids.append(dag_graph.edge_ids(in_cycle, out_cycle))
            #if count_cycles > we keep this as an option if counting takes too long

            count_cycles += 1

            if count_cycles > 10000:
                print('partial cull triggered')
                break #this is in case there are so many cycles, that even just counting them could take too long. in this case, we just do a partial cull and then count again

        if count_cycles > 100:#get rid of
            print(time.time() - start)
        #print(count_cycles, scene_idxs, flush=True)
        
        if count_cycles > 100:
            start = time.time()
            print(count_cycles)
            # dirty cycle reduction for when johnsons algorithm would take way too long (generally bogs down when count_cycles > 10000)
            # dirty cycle cutting just breaks all cycles by removing their least likely edge without checking if the removal of one edge might make the removal of another unnescesary. 
            # its not pretty, but if we have that many cycles, then this scene is in a state that is already kind of useless for learning
            to_remove = []
            for eid in eids:
                edge_probs_cycle = dag_graph.edata["edge_probs"][eid]
                remove_eid = eid[torch.argmin(edge_probs_cycle)]  
                to_remove.append(remove_eid)
            
            to_remove = torch.unique(torch.tensor(to_remove))
            #print(to_remove)#get rid
            dag_graph.remove_edges(to_remove.to(dag_graph.device))
            print(time.time() - start)#get rid
        else:
            break #if the number of cycles remaining is reasonable, we break.
        

    to_remove = []
    while len(eids) > 0:
        edge_probs_cycle = dag_graph.edata["edge_probs"][eids[0]]
        remove_eid = eids[0][torch.argmin(edge_probs_cycle)]
        to_remove.append(remove_eid)

        eids.pop(0)
        to_pop = []
        for j, eid_cycle in enumerate(eids):
            if remove_eid in eid_cycle:
                to_pop.append(j)
        
        eids = [v for i, v in enumerate(eids) if i not in to_pop]

    dag_graph.remove_edges(to_remove)

    return dag_graph

def build_dag_graph_test(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

    return dag_graph
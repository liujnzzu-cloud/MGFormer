import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

""" Build the user-agnostic global trajectory flow map from the sequence data """

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['MMSI'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)  #创建一个 带有进度条的迭代器
    for MMSI in loop:
        user_df = df[df['MMSI'] == MMSI]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['region_sequence']
            if node not in G.nodes():
                G.add_node(row['region_sequence'],
                           checkin_cnt=1,
                           Tanker_ratio=row['Tanker_ratio'],
                           Cargo_ratio=row['Cargo_ratio'],
                           latitude=row['latitude'],
                           longitude=row['longitude'],
                           area_m2=row['area_m2'],
                           )
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_region_sequence = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            region_sequence = row['region_sequence']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_region_sequence == 0) or (previous_traj_id != traj_id):
                previous_region_sequence = region_sequence
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_region_sequence, region_sequence):
                G.edges[previous_region_sequence, region_sequence]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_region_sequence, region_sequence, weight=1)
            previous_traj_id = traj_id
            previous_region_sequence = region_sequence

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/region_sequence, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w', encoding='utf-8') as f:
        print('region_sequence,checkin_cnt,Tanker_ratio,Cargo_ratio,area_m2,latitude,longitude', file=f)
        for each in nodes_data:
            region_sequence = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            Tanker_ratio = each[1]['Tanker_ratio']
            Cargo_ratio = each[1]['Cargo_ratio']
            area_m2 = each[1]['area_m2']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{region_sequence},{checkin_cnt},'
                  f'{Tanker_ratio},{Cargo_ratio},{area_m2},'
                  f'{latitude},{longitude}', file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude',feature5='borough'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4,feature5]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")




if __name__ == '__main__':
    dst_dir = r'dataset/AIS'


    df = pd.read_csv(os.path.join(dst_dir, 'trajectory_region_sequence_with_stats.csv'))
    df['region_sequence'] = df['region_sequence'].astype(str)
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(df)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)

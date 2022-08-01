import os
import pandas as pd
from data_structures import Cell, CellGraphCreator
import networkx as nx
import matplotlib.pyplot as plt
import dgl

output_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\Alex_summer_internship\\output"
data_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\Alex_summer_internship\\data"
data_fn = os.path.join(data_dir, "AllBMS_Tier1_FOVs.csv")

# data_header = "Centroid_X_um,Centroid_Y_um,Sample,Slide,Pathology,PD1_CellClassification,MorviusTeir1,Response".split(",")

# read data from csv file
df = pd.read_csv(data_fn)
all_sample_IDs = sorted(set(df["Sample"]))
print(all_sample_IDs)

color_dict = {"Tumor Cells": 'r', "Stromal Cells": 'b', "Immune Cells": 'g'}


def cell_graph_visualization(cell_graph, save_to=None, highlight_cells=None, node_colors=None, node_pos=None):
    cg = dgl.to_networkx(cell_graph)
    node_cnt = cell_graph.number_of_nodes()

    color_dict = {"Tumor Cells": 'r', "Stromal Cells": 'b', "Immune Cells": 'g'}

    plt.figure(1, figsize=[32, 32])
    if node_pos is None:
        node_pos = nx.nx_agraph.graphviz_layout(cell_graph)   # TODO: get cell locations
    if node_colors is None:
        if highlight_cells:
            node_colors = ['0.3'] * node_cnt
        else:
            node_colors = []
            for idx, c_x in enumerate(node_pos):
                cell_label = cell_graph.nodes[idx].label_txt
                if isinstance(cell_label, str):
                    cl = color_dict[cell_label]
                else:
                    cl = 'c'
                node_colors.append(cl)

    nx.draw_networkx(cg, node_color=node_colors, with_labels=False, pos=node_pos, node_size=2)
    plt.gca().invert_yaxis()
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


# create and save cell graph for each sample
for s_id in all_sample_IDs[0:10]:
    print("Processing %s" % s_id)
    sample_cells_idx = df["Sample"] == s_id
    sample_label = set(df["Pathology"][sample_cells_idx])
    sample_cells_x = list(df["Centroid_X_um"][sample_cells_idx])
    sample_cells_y = list(df["Centroid_Y_um"][sample_cells_idx])

    sample_cells_PDL1 = list(df["PD1_CellClassification"][sample_cells_idx])
    sample_cells_class = list(df["MorviusTeir1"][sample_cells_idx])

    print("%d nodes in the cell graph" % len(sample_cells_x))

    cg_fn = os.path.join(output_dir, s_id + "_graph.gml")

    print("Load saved cell graph for feature extraction")

    cg = nx.read_gpickle(cg_fn)
    print(cg.number_of_nodes())
    a = cg.nodes
    a.loc
    print(a[0])


    # TODO: extract features for the entire graph

    # TODO: sample the graph as sub-graphs, extract features

    # TODO: Are those features differentiable in disease subtypes?

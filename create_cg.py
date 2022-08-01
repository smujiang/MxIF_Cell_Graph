import os
import pickle

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from data_structures import Cell, CellGraphCreator

output_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\Alex_summer_internship\\output"
data_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\Alex_summer_internship\\data"
data_fn = os.path.join(data_dir, "AllBMS_Tier1_FOVs.csv")

# data_header = "Centroid_X_um,Centroid_Y_um,Sample,Slide,Pathology,PD1_CellClassification,MorviusTeir1,Response".split(",")

# read data from csv file
df = pd.read_csv(data_fn, engine='python')
all_sample_IDs = sorted(set(df["Sample"]))
print(all_sample_IDs)

color_dict = {"Tumor Cells": 'r', "Stromal Cells": 'b', "Immune Cells": 'g'}

# create and save cell graph for each sample
for s_id in all_sample_IDs[0:10]:
    s_id = "Mel30_BMS_region_003" # for debug
    print("Processing %s" % s_id)
    sample_cells_idx = df["Sample"] == s_id
    sample_label = set(df["Pathology"][sample_cells_idx])
    sample_cells_x = list(df["Centroid_X_um"][sample_cells_idx])
    sample_cells_y = list(df["Centroid_Y_um"][sample_cells_idx])

    sample_cells_PDL1 = list(df["PD1_CellClassification"][sample_cells_idx])
    sample_cells_class = list(df["MorviusTeir1"][sample_cells_idx])

    print("\t %d nodes in the cell graph" % len(sample_cells_x))

    # cell_all_PDL1_set = set(df["PD1_CellClassification"][sample_cells_idx])
    # cell_all_class_set = set(df["MorviusTeir1"][sample_cells_idx])

    # create cell graph
    cells = []
    pos = []
    col_map = []
    for idx, c_x in enumerate(sample_cells_x):
        c_y = sample_cells_y[idx]
        cell_label = sample_cells_class[idx]
        cell = Cell([c_x, c_y], s_id, label_txt=cell_label)
        cells.append(cell)
        pos.append([c_x, c_y])

        if isinstance(cell_label, str):
            cl = color_dict[cell_label]
        else:
            cl = 'c'
        col_map.append(cl)

    cg_creator = CellGraphCreator(cells, self_loop=False)
    cg = cg_creator.graph

    # save graph to output_dir
    # save_to = os.path.join(output_dir, s_id + "_graph.pickle")
    save_to = os.path.join(output_dir, s_id + "_graph.gml")
    if not os.path.exists(save_to):
        nx.write_gpickle(cg, save_to, protocol=2)
        # nx.write_gml(cg, save_to)

    # add legend and save figure to output_dir
    cell_graph = dgl.to_networkx(cg)
    plt.figure(1, figsize=[32, 32])
    nx.draw_networkx(cell_graph, node_color=col_map, with_labels=False, pos=pos, node_size=2)
    plt.gca().invert_yaxis()
    # plt.show()
    save_to = os.path.join(output_dir, s_id + "_graph.png")
    if not os.path.exists(save_to):
        plt.savefig(save_to)



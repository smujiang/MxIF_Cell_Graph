import os
import pickle
import numpy as np
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

DEBUG = True

def get_cell_count(cell_list, cell_label_txt="Tumor"):
    idx_list = []
    cnt = 0
    for idx, cell in enumerate(cell_list):
        if cell_label_txt in cell.label_txt:
            cnt += 1
            idx_list.append(idx)
    return cnt, idx_list

def simp_lable(label_txt):
    if "tumor" in label_txt.lower():
        return "tumor"
    elif "stroma" in label_txt.lower():
        return "stroma"
    else:
        return "immune"

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
    cell_labels = []
    for idx, c_x in enumerate(sample_cells_x):
        c_y = sample_cells_y[idx]
        cell_label = sample_cells_class[idx]
        cell_labels.append(simp_lable(cell_label))
        cell = Cell([c_x, c_y], s_id, label_txt=cell_label)
        cells.append(cell)
        pos.append([c_x, c_y])

        if isinstance(cell_label, str):
            cl = color_dict[cell_label]
        else:
            cl = 'c'
        col_map.append(cl)

    a = np.unique(pos, axis=0)

    cg_creator = CellGraphCreator(cells, self_loop=False)
    cg = cg_creator.graph

    tumor_cell_cnt, tumor_idx_list = get_cell_count(cells, "Tumor")
    stroma_cell_cnt, stroma_idx_list = get_cell_count(cells, "Stroma")
    immune_cell_cnt, immune_idx_list = get_cell_count(cells, "Immune")

    degrees = cg.degree()
    tumor_degrees_tmp = np.array(degrees)[tumor_idx_list]
    stroma_degrees_tmp = np.array(degrees)[stroma_idx_list]
    immune_degrees_tmp = np.array(degrees)[immune_idx_list]

    tumor_degrees = [i[1] for i in tumor_degrees_tmp]
    stroma_degrees = [i[1] for i in stroma_degrees_tmp]
    immune_degrees = [i[1] for i in immune_degrees_tmp]
    max_t_tmp = max(tumor_degrees)
    max_s_tmp = max(stroma_degrees)
    max_i_tmp = max(immune_degrees)

    tumor_deg_hist, _ = np.histogram(tumor_degrees, bins=10, range=(0, 20), density=True)
    stroma_deg_hist, _ = np.histogram(stroma_degrees, bins=10, range=(0, 20), density=True)
    immune_deg_hist, _ = np.histogram(immune_degrees, bins=10, range=(0, 20), density=True)

    new_cell_graph = cg.copy()
    new_pos = pos.copy()
    c_map_idx_remove = []
    for gn in range(len(cells)):
        neighbors = new_cell_graph.neighbors(gn)
        central_label = cell_labels[gn]
        neighbors_labels = []
        for n in neighbors:
            neighbors_labels.append(cell_labels[n])
        set_nl = list(set(neighbors_labels))
        if len(set_nl) == 0:
            new_cell_graph.remove_node(gn)
            c_map_idx_remove.append(gn)
            new_pos.pop(gn)
        if len(set_nl) == 1 and set_nl[0] == central_label:
            new_cell_graph.remove_node(gn)
            c_map_idx_remove.append(gn)
            new_pos.pop(gn)
    if DEBUG:
        c_map = [v for i, v in enumerate(col_map) if i not in c_map_idx_remove]

        plt.figure(2, figsize=[12, 12])
        # nx.draw(new_cell_graph, node_color=c_map, pos=new_pos)
        nx.draw(cg, node_color=col_map, pos=pos)
        p1_x = np.array(np.array(pos)[:, 0])[list(new_cell_graph.nodes)]
        p1_y = np.array(np.array(pos)[:, 1])[list(new_cell_graph.nodes)]
        plt.plot(p1_x, p1_y, "o", markersize=22)
        plt.gca().invert_yaxis()
        plt.show()

    entangle_cells_cnt = len(list(new_cell_graph.nodes))


    ROI_graph_features = [float(max_t_tmp), float(max_s_tmp), float(max_i_tmp),
                          float(entangle_cells_cnt) / float(tumor_cell_cnt + stroma_cell_cnt + immune_cell_cnt),
                          float(tumor_cell_cnt) / float(tumor_cell_cnt + stroma_cell_cnt + immune_cell_cnt),
                          float(stroma_cell_cnt) / float(tumor_cell_cnt + stroma_cell_cnt + immune_cell_cnt),
                          float(immune_cell_cnt) / float(tumor_cell_cnt + stroma_cell_cnt + immune_cell_cnt)]
    ROI_graph_features += list(tumor_deg_hist.astype(np.float))
    ROI_graph_features += list(stroma_deg_hist.astype(np.float))
    ROI_graph_features += list(immune_deg_hist.astype(np.float))


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



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

    cg_fn = os.path.join(output_dir, s_id + "_graph.pickle")

    # TODO: load saved cell graph for feature extraction
    print("TODO: load saved cell graph for feature extraction")



















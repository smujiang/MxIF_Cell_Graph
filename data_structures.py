import numpy as np
import dgl
from math import sqrt
from sklearn.metrics import pairwise_distances


class Cell:
    def __init__(self, loc, sample_id, features=None, label_id=-1, label_txt=""):
        self.sample_id = sample_id
        self.loc = np.array(loc)
        self.features = features
        self.label_id = label_id
        self.label_txt = label_txt


class CellGraphCreator:
    def __init__(self, cells=None, cell_graph_save_dir=None, distance=45, self_loop=True):
        self.start_idx = -1
        dm = self.get_connections(cells, distance, self_loop)
        if cells is not None:
            U, V = self.create_connections(cells, distance, self_loop)
            # self.graph = dgl.DGLGraph((U, V))
            # self.graph = dgl.DGLGraph((U, V))
            self.graph = dgl.graph((U, V))
        else:
            self.graph = self.load(cell_graph_save_dir)


    def get_connections(self, cells, distance, self_loop):
        coord = []
        for c in cells:
            coord.append(c.loc)
        dm = pairwise_distances(np.array(coord))
        return dm


    #TODO: different way to calculate connection matrix
    def create_connections(self, cells, distance, self_loop):
        U = []
        V = []
        for idx, cs in enumerate(cells):
            coord = cs.loc
            for idx_idx, ct in enumerate(cells):
                cd_cd = ct.loc
                if not idx_idx == idx:
                    dis = sqrt((coord[0] - cd_cd[0]) ** 2 + (coord[1] - cd_cd[1]) ** 2)
                    if dis < distance:
                        U.append(idx)
                        V.append(idx_idx)
                        if self.start_idx < 0:
                            self.start_idx = idx
                else:
                    if self_loop:
                        U.append(idx)
                        V.append(idx)
                        if self.start_idx < 0:
                            self.start_idx = idx
        return np.array(U)-self.start_idx, np.array(V)-self.start_idx

    def save(self, save_dir):
        print("save graph")

    def load(self, cg_dir):
        print("load graph")
        return None

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

class NetworkBuilder() :

    def __init__(self,path_to_data = 'raw/ImmuneXpressoResults.csv'):
        self._path_to_raw = path_to_data
        self._cells_edges = None

    def build_networkx_graph(self,enrichment_trh = 0):
        ix_df = pd.read_csv(self._path_to_raw)
        ix_df = ix_df[ix_df[" Enrichment Score"] > enrichment_trh]

        cell_to_cyto_df = ix_df[ix_df[" Actor"] == "cell"].drop(columns=[" NumPapers"," Enrichment Score","Cell Ontology ID"," Actor"," Action Sentiment"])
        cyto_to_cell_df = ix_df[ix_df[" Actor"] == "cytokine"].drop(columns=[" NumPapers"," Enrichment Score","Cell Ontology ID"," Actor"])

        cell_to_cell_df = cell_to_cyto_df.merge(cyto_to_cell_df,on = [" Cytokine Ontology Label"],suffixes  = ("_from","_to"))

        return nx.from_pandas_edgelist(cell_to_cell_df, ' Cell Ontology Label_from', ' Cell Ontology Label_to',edge_attr =" Action Sentiment",
                                    create_using=nx.DiGraph())

    def _show_graph(self,G):
        pos = nx.layout.circular_layout(G)
        nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True, pos=pos)
        plt.show()

        # node_sizes = [3 + 10 * i for i in range(len(G))]
        # M = G.number_of_edges()
        # edge_colors = range(2, M + 2)
        # edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        #
        # labels = {node: node for node in G.nodes()}
        # nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue',label=labels)
        # edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
        #                                arrowsize=10, edge_color=edge_colors,
        #                                edge_cmap=plt.cm.Blues, width=2)
        #
        # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
        # pc.set_array(edge_colors)
        # plt.colorbar(pc)
        #
        # ax = plt.gca()
        # ax.set_axis_off()
        # plt.show()




if __name__ == '__main__':
    nb = NetworkBuilder()
    G = nb.build_networkx_graph()
    nb._show_graph(G)

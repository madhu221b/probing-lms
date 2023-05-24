import torch
from ete3 import Tree as EteTree
from scipy.sparse.csgraph import minimum_spanning_tree

class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)
        
    def __str__(self):
        return self.get_ascii(show_internal=True)
    
    def __repr__(self):
        return str(self)



def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx
    
def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)
    return FancyTree(f"{newick_str};")

def create_gold_distances(corpus):
    all_distances = []

    for item in (corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)
 
        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))
        for src_node in range(1, sen_len+1):
            for target_node in range(1, sen_len+1):
                if distances[src_node-1][target_node-1] == 0:
                     s = ete_tree&str(src_node)
                     t  = ete_tree&str(target_node)
                     dist = s.get_distance(t)
                     distances[src_node-1][target_node-1] = dist
                     distances[target_node-1][src_node-1] = dist

        # Your code for computing all the distances comes here.
        all_distances.append(distances)

    return all_distances


def edges(mst):
    edges = set()
    edges_list = []
    for i, row in enumerate(mst):
        for j, val in enumerate(mst[i]):
            if int(val) == 1 and i<j: # i < j ensures that (1,2) is added and not (2,1) undirectional case
                edges_list.append((i,j))
                
    edges = set(edges_list)
    return edges

def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.
    return mst


def calc_uuas(pred_distances, gold_distances):
    num, denom = 0, 0
    uuas = 0
    
    for pred_matrix, gold_matrix in zip(pred_distances, gold_distances):
        pred_mst = create_mst(pred_matrix.to(torch.device('cpu')))
        gold_mst = create_mst(gold_matrix.to(torch.device('cpu')))
        pred_edges = edges(pred_mst)
        gold_edges = edges(gold_mst)
        edges_pg = pred_edges & gold_edges
        num += len(edges_pg)
        denom += len(gold_edges)

    uuas = num/denom
    return uuas
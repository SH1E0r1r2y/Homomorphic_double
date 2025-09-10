import uuid
from package.Homo.paillier import Paillier

def gen_id() -> str:
    return str(uuid.uuid4())[:8]

class IndexTreeNode:
    """實作 index tree"""
    def __init__(self, node_id, index_vector, left=None, right=None, fid=None):
        self.id = node_id
        self.P = index_vector    # 加密向量 (list of ciphertext tuples)
        self.Pl = left           # 左子節點
        self.Pr = right          # 右子節點
        self.fid = fid           # 葉節點才有

def build_index_tree(leaf_nodes, n2):
    nodes = leaf_nodes.copy()
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1] if i+1 < len(nodes) else None
            if right:
                parent_vector = [
                    Paillier.homomorphic_add(left.P[j], right.P[j], n2)
                    for j in range(len(left.P))
                ]
                parent = IndexTreeNode(
                    node_id=gen_id(),
                    index_vector=parent_vector,
                    left=left,
                    right=right,
                    fid=None
                )
                new_nodes.append(parent)
            else:
                new_nodes.append(left)
        nodes = new_nodes
    return nodes[0]  # 回傳樹根

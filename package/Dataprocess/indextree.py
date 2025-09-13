# ---- 加密索引樹節點 ----
class LeafNode:
    def __init__(self, doc_id, enc_vector):
        self.doc_id = doc_id
        self.enc_vector = enc_vector  # list of ciphertext tuples [(c1,c2), ...]
        self.next_leaf = None

class InternalNode:
    def __init__(self, children=None):
        self.children = children or []
        self.enc_vector = None  # 合併子節點向量

def homomorphic_sum(paillier, enc_vectors: list):
    """同態加法合併多個向量 => 放入Paillier?"""
    if not enc_vectors:
        return []
    summed = enc_vectors[0]
    for vec in enc_vectors[1:]:
        new_sum = [
            paillier.homomorphic_add(c1_tuple, c2_tuple, paillier.n2)
            for c1_tuple, c2_tuple in zip(summed, vec)
        ]
        summed = new_sum
    return summed


def build_index_tree(paillier, doc_blocks):
    # 建立葉節點
    leaves = [LeafNode(doc["doc_id"], doc["enc_tf"]) for doc in doc_blocks]

    current_level = leaves
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            children = current_level[i:i+2]
            enc_vec = homomorphic_sum(paillier, [c.enc_vector for c in children])
            node = InternalNode(children)
            node.enc_vector = enc_vec
            next_level.append(node)
        current_level = next_level
    root = current_level[0]
    return root
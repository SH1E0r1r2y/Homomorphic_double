# ---- 加密索引樹節點 ----
class Node:
    def __init__(self, enc_vector, left=None, right=None):
        self.enc_vector = enc_vector  # 加密向量 (EI' 或 P_all)
        self.left = left
        self.right = right
        self.is_leaf = False  # 預設為內部節點

class LeafNode(Node):
    def __init__(self, doc, enc_vector):
        super().__init__(enc_vector)
        self.doc = doc  # 存儲文檔元數據
        self.is_leaf = True
        self.enc_tf = doc["enc_tf"]

class InternalNode(Node):
    def __init__(self, left=None, right=None, enc_vector=None):
        super().__init__(enc_vector, left, right)
        self.is_leaf = False

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
    leaves = [LeafNode(doc, doc["enc_presence"]) for doc in doc_blocks]

    current_level = leaves
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                left_child = current_level[i]
                right_child = current_level[i + 1]
            else:
                left_child = current_level[i]
                right_child = None
            children_vectors = [c.enc_vector for c in [left_child, right_child] if c]
            enc_vec = homomorphic_sum(paillier, children_vectors)
            node = InternalNode(left=left_child, right=right_child, enc_vector=enc_vec)
            next_level.append(node)
        current_level = next_level
    return current_level[0] if current_level else None
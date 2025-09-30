# ---- 加密索引樹節點 ----
class LeafNode:
    def __init__(self, block, enc_vector):
        self.block = block
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
    leaves = [LeafNode(doc, doc["enc_presence"]) for doc in doc_blocks]

    # print(f"總 leaf 節點數量: {len(leaves)}")
    # for i, leaf in enumerate(leaves):
    #     print(f"Leaf {i}: doc_id={leaf.block['doc_id']}, enc_vector={leaf.enc_vector}")

    current_level = leaves
    level = 0
    while len(current_level) > 1:
        print(f"=== Level {level} 節點數量: {len(current_level)} ===")
        next_level = []
        for i in range(0, len(current_level), 2):
            children = current_level[i:i+2]
            enc_vec = homomorphic_sum(
                paillier, [c.enc_vector for c in children]
            )
            node = InternalNode(children)
            node.enc_vector = enc_vec
            next_level.append(node)
            print(f"InternalNode: children doc_ids={[c.block['doc_id'] for c in children] if hasattr(children[0],'block') else 'Inner'}")
            #print(f"enc_vector={enc_vec}")
        current_level = next_level
        level += 1

    #print(f"=== 最終 root 節點 enc_vector: {current_level[0].enc_vector} ===")
    return current_level[0]

def get_all_leaves(node):
    if hasattr(node, "block"):  # 葉子節點有 block
        return [node]
    leaves = []
    for child in getattr(node, "children", []):
        leaves.extend(get_all_leaves(child))
    return leaves
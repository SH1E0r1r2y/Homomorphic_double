# package/Trapdoor/trapdoor.py
import itertools
from typing import List
import heapq

def generate_trapdoor(paillier, vocab_list, query_keywords, h):
    """生成加密的查詢陷門向量"""
    # 建立查詢向量
    query_vector = [1 if kw in query_keywords else 0 for kw in vocab_list]
    encrypted_trapdoor = []
    for val in query_vector:
        encrypted_trapdoor.append(paillier.encrypt(val, h))
    
    return encrypted_trapdoor


def encrypted_vector_match(paillier, enc_vector, trapdoor,init_enc):
    """
    利用 Paillier 同態加法（homomorphic_add）與同態乘法計算
    做文件密文向量與 Trapdoor 向量的匹配度（點積）。
    """
    total_enc = init_enc

    for (c1, c2), (t_c1, t_c2) in zip(enc_vector, trapdoor):
        # 解密 t_c1 得到 query 權重 (scale or 0)
        t_plain = paillier.strong_decrypt(t_c1)
        if t_plain != 0:
            # 同態乘法：E(m1) ⊗ E(m2) = E(m1 * m2)：E(doc_value) ^ t_plain = E(doc_value * t_plain)
            scaled = paillier.homomorphic_scalar_multiply((c1, c2), t_plain)
            # 累計匹配結果 E(sum)
            total_enc = paillier.homomorphic_add(total_enc, scaled, paillier.n2)

    return paillier.strong_decrypt(total_enc[0])

counter = itertools.count()

def gbfs_search(paillier, root, trapdoor, init_enc, top_k=3):
    heap = []
    heapq.heappush(heap, (
        -encrypted_vector_match(paillier, root.enc_vector, trapdoor, init_enc),
        next(counter),
        root
    ))
    results = []
    while heap and len(results) < top_k:
        score, _, node = heapq.heappop(heap)
        if hasattr(node, "block"):
            # 从 node.block 里直接拿 doc_id 和 CID
            doc_id = node.block["doc_id"]
            cid = node.block["CID"]
            results.append((doc_id, cid, -score))
        else:
            for child in node.children:
                child_score = encrypted_vector_match(
                    paillier, child.enc_vector, trapdoor, init_enc
                )
                heapq.heappush(heap, (-child_score, next(counter), child))
    return results  # List of (doc_id, cid, score)




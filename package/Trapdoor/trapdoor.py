# package/Trapdoor/trapdoor.py

from typing import List
import heapq

# paillier 是你的 Paillier 實例，必須支援 encrypt() 與 homomorphic_add()

# -------------------------------
# Trapdoor 生成
# -------------------------------
def generate_trapdoor(paillier, vocab: list, query_keywords: list, pk_sys, scale: int = 1):
    """
    將查詢關鍵字轉為加密向量 (Trapdoor)
    - vocab: 詞彙表
    - query_keywords: 查詢關鍵字列表
    - scale: 權重 scaling
    """
    query_vector = [scale if kw in query_keywords else 0 for kw in vocab]
    enc_vector = [paillier.encrypt(m, pk_sys) for m in query_vector]  # 使用系統公鑰
    return enc_vector

def encrypted_vector_match(paillier, enc_vector, trapdoor,init_enc):
    """
    利用 Paillier 同態加法（homomorphic_add）與同態乘法計算
    做文件密文向量與 Trapdoor 向量的匹配度（點積）。
    """
    # 初始密文 E(0)
    total_enc = init_enc

    for (c1, c2), (t_c1, t_c2) in zip(enc_vector, trapdoor):
        # 解密 t_c1 得到 query 权重 (scale or 0)
        t_plain = paillier.strong_decrypt(t_c1)
        if t_plain != 0:
            # 同態乘法：E(m1) ⊗ E(m2) = E(m1 * m2)：E(doc_value) ^ t_plain = E(doc_value * t_plain)
            scaled = paillier.homomorphic_scalar_multiply((c1, c2), t_plain)
            # 同態加法：累計匹配結果 E(sum)
            total_enc = paillier.homomorphic_add(total_enc, scaled, paillier.n2)

    # 解密得到真正匹配度
    return paillier.strong_decrypt(total_enc[0])


import itertools

# 全域計數器
counter = itertools.count()

def gbfs_search(paillier, root, trapdoor, init_enc, top_k=3):
    heap = []
    # 推入 root 時攜帶一個唯一遞增 ID
    heapq.heappush(heap, (-encrypted_vector_match(paillier, root.enc_vector, trapdoor, init_enc),
                          next(counter), root))
    results = []

    while heap and len(results) < top_k:
        score, _, node = heapq.heappop(heap)
        if hasattr(node, "doc_id"):
            results.append((node.doc_id, -score))
        else:
            for child in node.children:
                child_score = encrypted_vector_match(paillier, child.enc_vector, trapdoor, init_enc)
                heapq.heappush(heap, (-child_score, next(counter), child))

    return results

# def gbfs_search(paillier, root, trapdoor, init_enc,top_k=3):
#     """
#     Greedy Breadth-First Search 搜尋加密索引樹
#     - root: InternalNode
#     - trapdoor: 加密向量
#     - top_k: 返回前 k 個最匹配的文件
#     """
#     heap = []
#     heapq.heappush(heap, (-encrypted_vector_match(paillier, root.enc_vector, trapdoor,init_enc), root))
#     results = []

#     while heap and len(results) < top_k:
#         score, node = heapq.heappop(heap)
#         if hasattr(node, "doc_id"):  # LeafNode
#             results.append((node.doc_id, -score))
#         else:  # InternalNode
#             for child in node.children:
#                 child_score = encrypted_vector_match(paillier, child.enc_vector, trapdoor,init_enc)
#                 heapq.heappush(heap, (-child_score, child))

#     # 可選: 二次排序（匹配值 + TF-IDF 加權分數）
#     return results

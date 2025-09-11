# package/Trapdoor/trapdoor.py

from typing import List
import heapq

# 假設 LeafNode, InternalNode 已經在 dataprocessing.py 定義
# paillier 是你的 Paillier 實例，必須支援 encrypt() 與 homomorphic_add()

# -------------------------------
# Trapdoor 生成
# -------------------------------
def generate_trapdoor(paillier, vocab: list, query_keywords: list, pk_sys, scale: int = 100):
    """
    將查詢關鍵字轉為加密向量 (Trapdoor)
    - vocab: 詞彙表
    - query_keywords: 查詢關鍵字列表
    - scale: 權重 scaling
    """
    query_vector = [scale if kw in query_keywords else 0 for kw in vocab]
    enc_vector = [paillier.encrypt(m, pk_sys) for m in query_vector]  # 使用系統公鑰
    return enc_vector


# -------------------------------
# 加密向量匹配函式
# -------------------------------
def encrypted_vector_match(paillier, enc_vector, trapdoor):
    """
    計算加密向量與 trapdoor 的匹配值
    簡化版: 使用加密向量 c1 與 trapdoor 權重近似
    """
    score = 0
    for (c1, c2), t in zip(enc_vector, trapdoor):
        if t != 0:
            score += c1  # 可用 Paillier 同態運算改進，這裡暫作簡化
    return score

# -------------------------------
# GBFS 搜尋
# -------------------------------
def gbfs_search(paillier, root, trapdoor, top_k=3):
    """
    Greedy Breadth-First Search 搜尋加密索引樹
    - root: InternalNode
    - trapdoor: 加密向量
    - top_k: 返回前 k 個最匹配的文件
    """
    heap = []
    heapq.heappush(heap, (-encrypted_vector_match(paillier, root.enc_vector, trapdoor), root))
    results = []

    while heap and len(results) < top_k:
        score, node = heapq.heappop(heap)
        if hasattr(node, "doc_id"):  # LeafNode
            results.append((node.doc_id, -score))
        else:  # InternalNode
            for child in node.children:
                child_score = encrypted_vector_match(paillier, child.enc_vector, trapdoor)
                heapq.heappush(heap, (-child_score, child))

    # 可選: 二次排序（匹配值 + TF-IDF 加權分數）
    return results

# package/Trapdoor/trapdoor.py
import itertools
import random
from typing import List,Tuple
import heapq

def generate_trapdoor(paillier, vocab_list, query_keywords, sk_du):
    """生成加密的查詢陷門向量，符合論文第 2.2 節定義。
    
    paillier: Paillier 實例
    vocab_list: 詞彙表
    query_keywords: 查詢關鍵詞列表
    sk_du: 數據用戶的弱私鑰 θ_u
    """
    query_vector = [1 if kw in query_keywords else 0 for kw in vocab_list]
    encrypted_trapdoor = []
    for val in query_vector:
        r_u = random.randrange(1, paillier.n)  # 隨機數
        theta_u = sk_du  # 數據用戶的弱私鑰
        c1 = (pow(paillier.g_core, r_u * theta_u, paillier.n2) * 
              (1 + val * paillier.n)) % paillier.n2
        c2 = pow(paillier.g_core, r_u, paillier.n2)
        encrypted_trapdoor.append((c1, c2))
    return encrypted_trapdoor

def ciphertext_refresh(paillier, enc_pair: Tuple[int, int], theta: int) -> Tuple[int, int]:
    """實現論文的 CR 機制 (11)-(12)，刷新密文使用 theta。"""
    c1, c2 = enc_pair
    n2 = paillier.n2
    r_prime = random.randrange(1, paillier.n)
    h_r_prime = pow(paillier.h_ta, r_prime, n2)
    g_r_prime = pow(paillier.g_core, r_prime, n2)
    # 根據公式 (11): CR(C1) = C1 * h^r' * (C2)^θ
    cr_c1 = (c1 * h_r_prime * pow(c2, theta, n2)) % n2
    # 根據公式 (12): CR(C2) = C2 * g^r'
    cr_c2 = (c2 * g_r_prime) % n2
    return (cr_c1, cr_c2)

def encrypted_vector_match(paillier, enc_vector, trapdoor, theta):
    """對 enc_vector 和 trapdoor 向量逐一比對，返回 Diff 向量"""
    n2 = paillier.n2
    match_vector = []

    if len(enc_vector) != len(trapdoor):
        raise ValueError("enc_vector 與 trapdoor 維度不一致")

    for ei, et in zip(enc_vector, trapdoor):
        if not (isinstance(ei, tuple) and isinstance(et, tuple) and len(ei) == 2 and len(et) == 2):
            raise ValueError("enc_vector 與 trapdoor 必須都是 (c1,c2) 密文對")

        cr_ei = ciphertext_refresh(paillier, ei, theta)

        # Diff = EI * ET^{-1}
        inv_et1 = pow(et[0], -1, n2)
        inv_et2 = pow(et[1], -1, n2)

        match_c1 = (cr_ei[0] * inv_et1) % n2
        match_c2 = (cr_ei[1] * inv_et2) % n2

        match_vector.append((match_c1, match_c2))

    return match_vector

def decrypt_and_count_matches(paillier, match_vector: List[Tuple[int, int]], mode="presence") -> int:
    total_matches = 0
    N = paillier.n
    
    for c_1, c_2 in match_vector:
        decrypted_m = paillier.strong_decrypt((c_1, c_2))
        
        # 核心修正：判斷結果是否為負數溢出
        if decrypted_m > N / 2:
            # 如果解密結果超過 N 的一半，視為負數。
            # 這裡假設明文空間 m < N/2
            decrypted_m = decrypted_m - N
        
        if mode == "presence":
            # 在 Presence 模式下，m_EI - m_ET = 0 才匹配
            if decrypted_m == 0:
                total_matches += 1
        
        elif mode == "tf":
            # TF Score: m_TF - 1，如果 m_TF=0，則結果是 -1。
            # 必須先處理負數 (溢出)，再還原 +1。
            
            # 如果是 -1，經過溢出處理後，decrypted_m 應該是 -1
            # 還原: -1 + 1 = 0
            
            # 如果是正常的 TF-1 (例如 72-1=71)，則 decrypted_m=71
            # 還原: 71 + 1 = 72 (正確)
            
            # 由於 TF 值不可能為負，如果結果是負數，則分數應為 0。
            final_tf_score = decrypted_m + 1
            if final_tf_score < 0:
                final_tf_score = 0
                
            total_matches += final_tf_score
            
    return total_matches


counter = itertools.count()
def gbfs_search(paillier, root, trapdoor, theta, top_k=5, t_plain=None) -> List[Tuple[int, str, int]]:
    """
    對整棵索引樹進行 Greedy Breadth-First Search (GBFS) 兩輪排序搜索。
    只返回前 k 個文件的匹配度、CID 和 doc_id。
    """
    heap = []
    counter = itertools.count()

    # 第一輪：匹配排序 (Match Sort) - 使用 root.enc_vector (P_all)
    match_cipher_vector = encrypted_vector_match(paillier, root.enc_vector, trapdoor, theta)
    match_score = decrypt_and_count_matches(paillier, match_cipher_vector, mode="presence")
    heapq.heappush(heap, (-match_score, next(counter), root))

    results = []

    while heap and len(results) < top_k:
        neg_match_score, _, node = heapq.heappop(heap)

        if node.is_leaf:
            # 第二輪：得分排序 (使用 enc_tf 的對應位置)
            if t_plain is not None:
                trapdoor_index = t_plain.index(1) if 1 in t_plain else 0  # 找到 T_plain 中為 1 的索引
                # 使用 enc_tf 的對應位置進行匹配
                # 只對應該位置的 trapdoor 元素（長度為 1）
                et_single = [trapdoor[trapdoor_index]]
                score_cipher_vector = encrypted_vector_match(paillier, [node.enc_tf[trapdoor_index]], et_single, theta)
                print(f"[Debug] Node {node.doc['doc_id']} enc_tf[{trapdoor_index}]: {node.enc_tf[trapdoor_index]}")
                print(f"[Debug] Score cipher vector: {score_cipher_vector}")
                score = decrypt_and_count_matches(paillier, score_cipher_vector, mode="tf")
                print(f"[Debug] Decrypted score: {score}")
            else:
                score = 0  # 默認得分
            doc_info = (score, node.doc["CID"], node.doc["doc_id"])
            results.append(doc_info)
        else:
            for child in (node.left, node.right):
                if child:
                    child_cipher_vector = encrypted_vector_match(paillier, child.enc_vector, trapdoor, theta)
                    child_match_score = decrypt_and_count_matches(paillier, child_cipher_vector)
                    heapq.heappush(heap, (-child_match_score, next(counter), child))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
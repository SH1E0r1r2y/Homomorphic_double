# package/Trapdoor/trapdoor.py
import math
import time
import random
import itertools, heapq
from typing import List, Dict, Tuple, Any
import heapq
#from indextree import get_all_leaves

def generate_trapdoor(paillier, vocab_list, query_keywords, du, sk_du,top_k):
    """生成加密的查詢陷門向量"""
    query_vector = [1 if kw in query_keywords else 0 for kw in vocab_list]
    ET = []
    for val in query_vector:
        r = random.randint(1, paillier.n // 4)
        c1 = (pow(paillier.g_core, r * sk_du, paillier.n2) * ((1 + val * paillier.n) % paillier.n2)) % paillier.n2
        c2 = pow(paillier.g_core, r, paillier.n2)
        ET.append((c1, c2))
    # Warrant 這裡先簡單模擬成一個 token
    warrant = f"Warrant(DU={du.id})"
    return (warrant, ET, top_k)

def get_current_timestamp():
    return int(time.time())

def encrypted_vector_match(paillier, enc_vector, trapdoor):
    """
    在密文空間比較 CR(EI')*CR(ET) 與 CR(ET)
    - 匹配 (prod!=ET) 累加 E(1) 到密文計數器
    """
    # 必須用加密的 0 作為初始計數器
    E_match_count = paillier.encrypt(0, paillier.h_ta) 
    # 必須用加密的 1 作為匹配成功的累加值
    E_one = paillier.encrypt(1, paillier.h_ta) 

    for EI_ct, ET_ct in zip(enc_vector, trapdoor):
        if ET_ct is None:
            continue

        # 1. 同態乘法 (Paillier 加法)
        # 注意：你需要使用 Paillier 類中的 homomorphic_add 靜態方法
        prod_ct = paillier.homomorphic_add(EI_ct, ET_ct, paillier.n2)

        # 2. 密文比較 (檢查是否不等於 E(1))
        # 這是密文域的布林判斷，不能解密
        if prod_ct != ET_ct:
            # Match 成功 (明文結果 = 2, 不等於 1)
            # 累加 E(1) 到密文計數器
            E_match_count = paillier.homomorphic_add(E_match_count, E_one, paillier.n2)
            
    # 返回密文結果 E(Match Count)
    return E_match_count


# 必須在函數外部定義（作為佔位符）
def get_current_timestamp():
    return time.time()

def get_all_leaves(node):
    if hasattr(node, "block"):  # 葉子節點有 block
        return [node]
    leaves = []
    for child in getattr(node, "children", []):
        leaves.extend(get_all_leaves(child))
    return leaves

def gbfs_match_sort(paillier, root, T_plain, participants, seta_recov, lambda_recov, top_k) -> List[Tuple[Dict, int]]:
    """
    執行 GBFS 搜索，返回包含明文 Match Score 的候選文件列表。
    
    返回格式: [({result_info}, match_score_plain), ...]
    """
    # Step 1: 生成簡化陷門 E_T_cr
    E_T_cr = []
    for bit in T_plain:
        if bit == 1:
            ct = paillier.encrypt(1, paillier.h_ta)
            for fn in participants:
                ct = paillier.refresh(ct, seta_recov) 
            E_T_cr.append(ct)
        else:
            E_T_cr.append(None)

    # Step 2: 初始化 heap
    heap = []
    counter = itertools.count()

    # Step 3: 刷新根節點並計算分數
    E_root_vector = []
    for j in range(len(T_plain)):
        tmp_ct = root.enc_vector[j]
        for fn in participants:
            tmp_ct = paillier.refresh(tmp_ct, seta_recov)
        E_root_vector.append(tmp_ct)

    E_root_score = encrypted_vector_match(paillier, E_root_vector, E_T_cr)
    root_score_plain = paillier.strong_decrypt(E_root_score, lambda_recov) 
    heapq.heappush(heap, (-root_score_plain, next(counter), root))

    candidate_nodes = []
    current_time = get_current_timestamp()

    # Step 4: GBFS 搜索 (Match Sort)
    while heap and len(candidate_nodes) < 2 * top_k: # 篩選 2*top_k 個候選
        score_neg, _, node = heapq.heappop(heap)
        score_plain = -score_neg 

        if hasattr(node, "block"):  # Leaf node
            if hasattr(node, "deadline") and node.deadline < current_time:
                continue

            result_info = {
                'doc_id': node.block.get('doc_id'),
                'CID': node.block.get('CID'),
                'enc_tf': node.block.get('enc_tf'), 
                'enc_presence': node.block.get('enc_presence'),
            }
            candidate_nodes.append((result_info, score_plain)) 
        else:  # Inner node
            for child in node.children:
                E_child_vector = []
                for j in range(len(T_plain)):
                    tmp_ct = child.enc_vector[j]
                    for fn in participants:
                        tmp_ct = paillier.refresh(tmp_ct, seta_recov)
                    E_child_vector.append(tmp_ct)

                E_child_score = encrypted_vector_match(paillier, E_child_vector, E_T_cr)
                child_score_plain = paillier.strong_decrypt(E_child_score, lambda_recov)
                
                if child_score_plain > 0:
                    heapq.heappush(heap, (-child_score_plain, next(counter), child))

    return candidate_nodes

def score_and_rank(paillier, root, T_plain, lambda_recov, top_k, candidate_nodes: List[Tuple[Dict, int]]):
    """
    計算候選文件 (candidate_nodes) 的 TF-IDF 分數，並執行兩輪排序。
    
    返回格式: [{'ranking': 1, 'doc_id': 1, 'CID': '...'}, ...]
    """
    # 假設 math 已經被 import
    import math 
    
    vocab_size = len(T_plain)
    
    # 1. 計算全局 IDF 因子 (N_wj)
    leaf_nodes = get_all_leaves(root)
    F_total = len(leaf_nodes) 
    N_wj_plain = [0] * vocab_size
    
    # 計算包含關鍵字的文件數 N_wj
    for j in range(vocab_size):
        count = 0
        # 注意: 這裡的 leaf 應是 IndexTree 的 Node，包含 block 屬性
        for leaf in leaf_nodes: 
            # 獲取 enc_presence 密文，並使用 lambda_recov 強解密
            E_presence = leaf.block['enc_presence'][j]
            presence_dec = paillier.strong_decrypt(E_presence, lambda_recov) 

            if presence_dec == 1:
                count += 1
        N_wj_plain[j] = max(count, 1) 
        
    IDF = [math.log(F_total / N_wj_plain[j]) for j in range(vocab_size)]

    # 2. 計算 TF-IDF 分數並收集結果
    scored_results = []
    
    for result_info, match_score in candidate_nodes:
        
        # 獲取加密 TF 向量
        enc_tf_vector = result_info.get('enc_tf')
        
        # 強解密 TF 向量
        tf_vector_plain = [paillier.strong_decrypt(c, lambda_recov) for c in enc_tf_vector]
        
        # 計算 TF-IDF Score
        tfidf_score = sum(tf_vector_plain[j] * IDF[j] for j in range(vocab_size))
        
        scored_results.append({
            'doc_id': result_info.get('doc_id'),
            'CID': result_info.get('CID'),
            'match_score': match_score, 
            'tfidf_score': tfidf_score
        })

    # 3. 最終兩輪排序 (Match Sort + Score Sort)
    scored_results.sort(key=lambda x: (x['match_score'], x['tfidf_score']), reverse=True)
    
    # 4. 格式化輸出
    final_output = []
    for i, result in enumerate(scored_results[:top_k]):
        final_output.append({
            'doc_id': result['doc_id'],
            'CID': result['CID']
        })
        
    return final_output
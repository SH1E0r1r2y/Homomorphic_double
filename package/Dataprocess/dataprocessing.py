# package/Homo/dataprocessing.py
import random
import math
from collections import Counter
from typing import List, Tuple

# ---- 產生假資料 ----
def generate_corpus(num_docs: int = 5,
                    vocab_size: int = 20,
                    avg_terms_per_doc: int = 6,
                    seed: int | None = None) -> Tuple[List[str], List[List[str]]]:
    """
    模擬：
    - 回傳 (vocabulary_list, docs_as_list_of_tokens)
    - vocabulary_list: ['kw1','kw2',...]
    - docs_as_list_of_tokens: [['kw3','kw7',...], ...]
    """
    if seed is not None:
        random.seed(seed)

    # 建立詞彙表
    vocab = [f"kw{idx}" for idx in range(1, vocab_size + 1)]

    docs = []
    for _ in range(num_docs):
        # 每篇文件從 vocab 隨機抽取若干關鍵字（允許重複以模擬詞頻）
        m = max(1, int(random.gauss(avg_terms_per_doc, avg_terms_per_doc / 3)))
        tokens = [random.choice(vocab) for _ in range(m)]
        docs.append(tokens)

    return vocab, docs

def compute_tf_vectors(vocab, docs):
    tf_vectors = []
    for d in docs:
        doc_len = len(d)
        vec = []
        for w in vocab:
            count = d.count(w)
            tf = (1 + math.log(count)) / (1 + math.log(doc_len)) if count > 0 else 0
            vec.append(tf)
        tf_vectors.append(vec)
    return tf_vectors

def compute_presence_vectors(vocab, docs):
    pres_vectors = []
    for d in docs:
        vec = [1 if w in d else 0 for w in vocab]
        pres_vectors.append(vec)
    return pres_vectors

def compute_tfidf_vectors(tf_vectors, pres_vectors, scale=100):
    """
    帶除錯資訊的版本，方便驗證計算過程
    """
    num_docs = len(tf_vectors)
    vocab_size = len(tf_vectors[0])
    
    print(f"[DEBUG] 計算 TF-IDF，文件數: {num_docs}, 詞彙大小: {vocab_size}")
    print(f"[DEBUG] 原始 TF 向量: {tf_vectors}")
    print(f"[DEBUG] 出現向量: {pres_vectors}")
    
    # 計算 DF 和 IDF
    df = [0] * vocab_size
    for j in range(vocab_size):
        df[j] = sum(pres_vectors[i][j] for i in range(num_docs))
    
    idf = [math.log(num_docs / df[j]) if df[j] > 0 else 0 for j in range(vocab_size)]
    
    print(f"[DEBUG] 文件頻率 (DF): {df}")
    print(f"[DEBUG] 逆文件頻率 (IDF): {[round(x, 4) for x in idf]}")
    
    tfidf_float = []
    tfidf_int = []
    
    for i in range(num_docs):
        print(f"\n[DEBUG] 處理文件 {i+1}:")
        
        # 對數加權 TF
        weighted_tf = []
        for j in range(vocab_size):
            raw_count = tf_vectors[i][j]
            if raw_count > 0:
                weighted_tf_val = 1 + math.log(raw_count)
                print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val:.4f}")
            else:
                weighted_tf_val = 0
                print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val}")
            weighted_tf.append(weighted_tf_val)
        
        # TF-IDF
        tfidf_doc = [weighted_tf[j] * idf[j] for j in range(vocab_size)]
        print(f"  TF-IDF (未規範化): {[round(x, 4) for x in tfidf_doc]}")
        
        # L2 規範化
        l2_norm = math.sqrt(sum(val**2 for val in tfidf_doc))
        print(f"  L2 範數: {l2_norm:.4f}")
        
        if l2_norm > 0:
            normalized_tfidf = [val / l2_norm for val in tfidf_doc]
        else:
            normalized_tfidf = tfidf_doc
        
        print(f"  規範化後 TF-IDF: {[round(x, 4) for x in normalized_tfidf]}")
        
        # 整數化
        tfidf_int_doc = [int(val * scale) for val in normalized_tfidf]
        print(f"  整數化 TF-IDF (×{scale}): {tfidf_int_doc}")
        
        tfidf_float.append(normalized_tfidf)
        tfidf_int.append(tfidf_int_doc)
    
    return tfidf_float, tfidf_int

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
    """同態加法合併多個向量"""
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
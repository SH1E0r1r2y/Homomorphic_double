# package/Homo/dataprocessing.py
import random
import math
from cid import make_cid
import multihash
import hashlib
from typing import List, Tuple

def compute_local_cid(data: bytes) -> str:
    hash_digest = hashlib.sha256(data).digest()
    mh = multihash.digest(data, "sha2-256")
    mh_bytes = mh.encode()
    # 4) 建立 CID （CIDv1 + dag-pb codec）
    cid_obj = make_cid(1, "dag-pb", mh_bytes)
    return str(cid_obj)

def generate_corpus(num_docs: int,
                    vocab_size: int,
                    avg_terms_per_doc: int,
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

def compute_presence_vectors(vocab, docs):
    pres_vectors = []
    for d in docs:
        vec = [1 if w in d else 0 for w in vocab]
        pres_vectors.append(vec)
    return pres_vectors

def compute_raw_tf_vectors(vocab, docs):
    """
    計算原始詞頻計數向量（整數）
    
    Args:
        vocab: 詞彙表
        docs: 文件列表
        
    Returns:
        tf_vectors: 包含整數計數的 TF 向量
    """
    tf_vectors = []
    for doc in docs:
        tf_vector = []
        for word in vocab:
            count = doc.count(word)  # 計算詞在文件中出現的次數
            tf_vector.append(count)
        tf_vectors.append(tf_vector)
    return tf_vectors

def compute_tfidf_vectors(tf_vectors, pres_vectors, scale=100):
    """
    根據論文 4.2.1 節計算 TF-IDF 向量
    Args:
        tf_vectors: 原始詞頻計數向量 (整數計數)
        pres_vectors: 出現向量 (presence vectors)
        scale: 整數化縮放因子，設為 100
    
    Returns:
        tfidf_float: 浮點數 TF-IDF 向量（規範化後）
        tfidf_int: 整數 TF-IDF 向量（規範化並縮放後）
    """
    num_docs = len(tf_vectors)
    vocab_size = len(tf_vectors[0])
    
    df = [0] * vocab_size
    for j in range(vocab_size):
        df[j] = sum(pres_vectors[i][j] for i in range(num_docs))
    
    idf = [math.log(num_docs / df[j]) if df[j] > 0 else 0 for j in range(vocab_size)]
    
    #tfidf_float = []
    tfidf_int = []
    
    for i in range(num_docs):
        # 3.1 計算對數加權的 TF: TF_{i,j} = 1 + ln(N_{i,j})
        weighted_tf = []
        for j in range(vocab_size):
            raw_count = tf_vectors[i][j]
            if raw_count > 0:
                # 計算 TF_{i,j} = 1 + ln(N_{i,j})
                weighted_tf_val = 1 + math.log(raw_count)
                #print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val:.4f}")
            else:
                weighted_tf_val = 0
                #print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val}")
            weighted_tf.append(weighted_tf_val)
        
        # 3.2 計算 TF-IDF: TF-IDF_{i,j} = TF_{i,j} × IDF_j
        tfidf_doc = [weighted_tf[j] * idf[j] for j in range(vocab_size)]
        #print(f"  TF-IDF (未規範化): {[round(x, 4) for x in tfidf_doc]}")
        
        # 3.3 向量規範化 (L2 normalization)
        l2_norm = math.sqrt(sum(val**2 for val in tfidf_doc))
        #print(f"  L2 範數: {l2_norm:.4f}")
        
        if l2_norm > 0:
            normalized_tfidf = [val / l2_norm for val in tfidf_doc]
        else:
            normalized_tfidf = tfidf_doc
        
        #print(f"  規範化後 TF-IDF: {[round(x, 4) for x in normalized_tfidf]}")
        
        # 3.4 整數化處理 - 確保結果是正整數
        tfidf_int_doc = []
        for val in normalized_tfidf:
            scaled_val = int(abs(val) * scale)  # 取絕對值確保為正數
            tfidf_int_doc.append(scaled_val)
        
        #print(f"  整數化 TF-IDF (×{scale}): {tfidf_int_doc}")
        
        #tfidf_float.append(normalized_tfidf)
        tfidf_int.append(tfidf_int_doc)
    
    return tfidf_int

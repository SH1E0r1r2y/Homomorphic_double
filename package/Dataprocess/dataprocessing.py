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

def compute_tfidf_vectors(tf_vectors, pres_vectors):
    num_docs = len(tf_vectors)
    vocab_size = len(tf_vectors[0])
    df = [0] * vocab_size
    for j in range(vocab_size):
        df[j] = sum(pres_vectors[i][j] for i in range(num_docs))
    idf = [math.log(num_docs / df[j]) if df[j] > 0 else 0 for j in range(vocab_size)]
    
    tfidf_float = []
    tfidf_int = []
    for i in range(num_docs):
        tfidf_f = [tf_vectors[i][j] * idf[j] for j in range(vocab_size)]
        tfidf_int_i = [int(tfidf_f[j] * 100) for j in range(vocab_size)]  # scaling
        tfidf_float.append(tfidf_f)
        tfidf_int.append(tfidf_int_i)
    return tfidf_float, tfidf_int
# package/Homo/dataprocessing.py
import random
import math
from collections import Counter
from typing import List, Tuple

# ---- 產生模擬資料 ----
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

# ---- 建立 TF 向量（單純 term frequency） ----
def compute_tf_vectors(vocab: List[str], docs: List[List[str]]) -> List[List[int]]:
    """
    回傳 TF 向量（整數計數），shape = (num_docs, vocab_len)
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    vectors = []
    for tokens in docs:
        ctr = Counter(tokens)
        vec = [0] * len(vocab)
        for tok, cnt in ctr.items():
            vec[vocab_index[tok]] = cnt
        vectors.append(vec)
    return vectors

# ---- 建立 binary presence 向量 ----
def compute_presence_vectors(vocab: List[str], docs: List[List[str]]) -> List[List[int]]:
    vocab_index = {w: i for i, w in enumerate(vocab)}
    vectors = []
    for tokens in docs:
        vec = [0] * len(vocab)
        for tok in set(tokens):
            vec[vocab_index[tok]] = 1
        vectors.append(vec)
    return vectors

# ---- 建立 TF-IDF 向量（浮點數），會回傳整數化版本供加密用 ----
def compute_tfidf_vectors(vocab: List[str], docs: List[List[str]],
                          scale: int = 100) -> Tuple[List[List[float]], List[List[int]]]:
    """
    - scale: 乘上去後四捨五入為整數，因為 Paillier 只能加整數。
    - 回傳 (tfidf_float_vectors, tfidf_int_vectors)
    """
    N = len(docs)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    df = [0] * len(vocab)
    # document frequency
    for tokens in docs:
        seen = set(tokens)
        for w in seen:
            df[vocab_index[w]] += 1

    tfidf_float = []
    tfidf_int = []
    for tokens in docs:
        ctr = Counter(tokens)
        total_terms = len(tokens) if len(tokens) > 0 else 1
        vec_f = [0.0] * len(vocab)
        vec_i = [0] * len(vocab)
        for w, idx in vocab_index.items():
            tf = ctr[w] / total_terms if total_terms > 0 else 0.0
            idf = math.log((N + 1) / (df[idx] + 1)) + 1.0  # smoothed idf
            val = tf * idf
            vec_f[idx] = val
            # scale → int (note: rounding; choose scale to control precision)
            vec_i[idx] = int(round(val * scale))
        tfidf_float.append(vec_f)
        tfidf_int.append(vec_i)
    return tfidf_float, tfidf_int

# ---- Paillier 加密向量（每一維加密成 ciphertext tuple (C1,C2)） ----
def encrypt_vector(paillier, h: int, vector: List[int]) -> List[Tuple[int,int]]:
    """
    - paillier: Paillier instance
    - h: entity public h (from gen_entity_key)
    - vector: list of integers (must be 0 <= x < N). If some values exceed N, caller should reduce mod N.
    - 回傳 list of ciphertext tuples
    """
    enc = []
    for val in vector:
        m = val % paillier.n
        c1, c2 = paillier.encrypt(m, h=h)
        enc.append((c1, c2))
    return enc

# ---- 還原（強解密）向量 ----
def decrypt_vector_with_strong(paillier, enc_vector):
    res = []
    for (c1, _c2) in enc_vector:
        m = paillier.strong_decrypt(c1)
        # 強制還原到合法範圍
        m = m % paillier.n
        res.append(m)
    return res


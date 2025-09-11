import json
import math
from package.Homo.paillier import Paillier
from package.Dataprocess.dataprocessing import generate_corpus, compute_presence_vectors, build_index_tree
#from package.Trapdoor.trapdoor import generate_trapdoor, gbfs_search

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

# def compute_tfidf_vectors(tf_vectors, pres_vectors, scale=100):
#     """
#     根據論文 4.2.1 節計算 TF-IDF 向量
    
#     Args:
#         tf_vectors: 原始詞頻計數向量 (整數計數)
#         pres_vectors: 出現向量 (presence vectors)
#         scale: 整數化縮放因子，預設為 100
    
#     Returns:
#         tfidf_float: 浮點數 TF-IDF 向量（規範化後）
#         tfidf_int: 整數 TF-IDF 向量（規範化並縮放後）
#     """
#     num_docs = len(tf_vectors)
#     vocab_size = len(tf_vectors[0])
    
#     print(f"[INFO] 計算 TF-IDF，文件數: {num_docs}, 詞彙大小: {vocab_size}")
#     print(f"[INFO] 原始 TF 向量: {tf_vectors}")
    
#     # 1. 計算文件頻率 (DF - Document Frequency)
#     df = [0] * vocab_size
#     for j in range(vocab_size):
#         df[j] = sum(pres_vectors[i][j] for i in range(num_docs))
    
#     # 2. 計算逆文件頻率 (IDF - Inverse Document Frequency)
#     idf = [math.log(num_docs / df[j]) if df[j] > 0 else 0 for j in range(vocab_size)]
    
#     print(f"[INFO] 文件頻率 (DF): {df}")
#     print(f"[INFO] 逆文件頻率 (IDF): {[round(x, 4) for x in idf]}")
    
#     # 3. 計算加權 TF 和 TF-IDF
#     tfidf_float = []
#     tfidf_int = []
    
#     for i in range(num_docs):
#         print(f"\n[INFO] 處理文件 {i+1}:")
        
#         # 3.1 計算對數加權的 TF: TF_{i,j} = 1 + ln(N_{i,j})
#         weighted_tf = []
#         for j in range(vocab_size):
#             raw_count = tf_vectors[i][j]
#             if raw_count > 0:
#                 # 論文公式: TF_{i,j} = 1 + ln(N_{i,j})
#                 weighted_tf_val = 1 + math.log(raw_count)
#                 print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val:.4f}")
#             else:
#                 weighted_tf_val = 0
#                 print(f"  詞 {j}: 原始計數={raw_count}, 加權TF={weighted_tf_val}")
#             weighted_tf.append(weighted_tf_val)
        
#         # 3.2 計算 TF-IDF: TF-IDF_{i,j} = TF_{i,j} × IDF_j
#         tfidf_doc = [weighted_tf[j] * idf[j] for j in range(vocab_size)]
#         print(f"  TF-IDF (未規範化): {[round(x, 4) for x in tfidf_doc]}")
        
#         # 3.3 向量規範化 (L2 normalization)
#         l2_norm = math.sqrt(sum(val**2 for val in tfidf_doc))
#         print(f"  L2 範數: {l2_norm:.4f}")
        
#         if l2_norm > 0:
#             normalized_tfidf = [val / l2_norm for val in tfidf_doc]
#         else:
#             normalized_tfidf = tfidf_doc
        
#         print(f"  規範化後 TF-IDF: {[round(x, 4) for x in normalized_tfidf]}")
        
#         # 3.4 整數化處理 - 確保結果是正整數
#         tfidf_int_doc = []
#         for val in normalized_tfidf:
#             scaled_val = int(abs(val) * scale)  # 取絕對值確保為正數
#             tfidf_int_doc.append(scaled_val)
        
#         print(f"  整數化 TF-IDF (×{scale}): {tfidf_int_doc}")
        
#         tfidf_float.append(normalized_tfidf)
#         tfidf_int.append(tfidf_int_doc)
    
#     return tfidf_float, tfidf_int

def demo():
    # 1) 產生系統金鑰與 DO 公鑰
    paillier = Paillier.keygen(k=64)
    do_entity = paillier.gen_entity_key()
    sys_entity = paillier.gen_entity_key()  # 系統公鑰 pk_TA

    n_do, g_do, h_do = do_entity["pk"]
    n_sys, g_sys, h_sys = sys_entity["pk"]

    # 2) 模擬資料
    vocab, docs = generate_corpus(num_docs=1, vocab_size=5, avg_terms_per_doc=8, seed=42)
    print(f"[INFO] 產生的詞彙表: {vocab}")
    print(f"[INFO] 產生的文件: {docs}")

    # 3) 計算原始整數 TF 向量
    tf_vectors = compute_raw_tf_vectors(vocab, docs)
    #pres_vectors = compute_presence_vectors(vocab, docs)
    
    print(f"\n[INFO] 原始整數 TF 向量: {tf_vectors}")
    #print(f"[INFO] 出現向量: {pres_vectors}")
    
    # # 使用修正後的 TF-IDF 計算
    # tfidf_float, tfidf_int = compute_tfidf_vectors(tf_vectors, pres_vectors)

    # 4) 加密 - 修正加密呼叫
    print(f"\n[INFO] 開始加密...")
    enc_tf = []
    for doc_idx, doc in enumerate(tf_vectors):
        print(f"處理文件 {doc_idx}: {doc}")
        enc_doc = []
        for val_idx, val in enumerate(doc):
            if not isinstance(val, int) or val < 0:
                print(f"[ERROR] TF 值必須是非負整數，得到: {val} (類型: {type(val)})")
                continue
            
            # 詳細記錄每次加密
            encrypted_pair = paillier.encrypt(val, h_do)
            print(f"  加密 val[{val_idx}]={val} -> c1={encrypted_pair[0]}")
            
            # 立即驗證解密
            dec_check = paillier.strong_decrypt(encrypted_pair[0])
            print(f"  立即解密檢查: {dec_check}")
            
            enc_doc.append(encrypted_pair)
        enc_tf.append(enc_doc)

    from sympy import isprime

    # 假设你能拿到 p 和 q
    print("p prime? ", isprime(paillier.p))
    print("q prime? ", isprime(paillier.q))
    print("p' prime? ", isprime((paillier.p-1)//2))
    print("q' prime? ", isprime((paillier.q-1)//2))
    assert paillier.n2 == paillier.n * paillier.n
    from math import lcm
    lam_true = lcm(paillier.p-1, paillier.q-1)
    print("lambda_dec correct? ", paillier.lambda_dec == lam_true)
    # 验证 g = n + 1
    print("g == n+1? ", paillier.g_core == paillier.n + 1)

    # 或者验证更通用条件：
    from package.Homo.utils import L
    u = pow(paillier.g_core, paillier.lambda_dec, paillier.n2)
    l_u = L(u, paillier.n)
    print("gcd(L(g^λ), n) == 1? ", math.gcd(l_u, paillier.n) == 1)


    
    # enc_pres = []
    # for doc_idx, doc in enumerate(pres_vectors):
    #     enc_doc = []
    #     for val in doc:
    #         encrypted_pair = paillier.encrypt(val, h_sys)
    #         enc_doc.append(encrypted_pair)
    #     enc_pres.append(enc_doc)
    
    # enc_tfidf = []
    # for doc_idx, doc in enumerate(tfidf_int):
    #     enc_doc = []
    #     for val in doc:
    #         if not isinstance(val, int) or val < 0:
    #             print(f"[ERROR] TF-IDF 值必須是非負整數，得到: {val} (類型: {type(val)})")
    #             continue
    #         encrypted_pair = paillier.encrypt(val, h_do)
    #         enc_doc.append(encrypted_pair)
    #     enc_tfidf.append(enc_doc)

    print(f"[INFO] 加密完成！")

    # 5) 驗證解密
    print(f"\n[INFO] 驗證解密...")
    dec_tf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tf]
    # dec_pres = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_pres]
    # dec_tfidf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tfidf]

    # 比對原始
    print("\n=== 解密驗證 ===")
    print("原始 TF:", tf_vectors[0])
    print("解密 TF:", dec_tf[0])
    print("=== 驗證解密 ===")

    # 單筆逐元素解密測試
    for i, (c1, c2) in enumerate(enc_tf[0]):
        val = paillier.strong_decrypt(c1)
        print(f"Enc[{i}]: c1={c1}, Dec={val}")
    print("=== 系統參數檢查 ===")
    print("n =", paillier.n)
    print("n2 =", paillier.n2) 
    print("lambda_dec =", paillier.lambda_dec)
    print("mu =", paillier.mu)

    print("\n=== 異常密文檢查 ===")
    # 檢查第一個文件第三個詞（索引2）的加密
    problem_cipher = enc_tf[0][2]  # (c1, c2)
    c1, c2 = problem_cipher
    print(f"異常密文: c1={c1}, c2={c2}")

    # 檢查 L 函數中間步驟
    test_val = pow(c1, paillier.lambda_dec, paillier.n2)
    print(f"pow(c1, lambda, n2) = {test_val}")

    # 手動計算L函數
    from package.Homo.utils import L
    l_result = L(test_val, paillier.n)
    print(f"L(test_val, n) = {l_result}")

    # 最終解密結果
    final_result = (l_result * paillier.mu) % paillier.n
    print(f"最終解密結果 = {final_result}")

    print("\n=== 單獨測試該值加解密 ===")
    # 單獨加解密測試原始值 1
    m = 1
    enc_test = paillier.encrypt(m, h_do)
    print(f"測試加密 {m}: {enc_test}")
    dec_test = paillier.strong_decrypt(enc_test[0])
    print(f"測試解密: {dec_test}")

    print("\n=== μ參數驗證 ===")
    # 重新計算μ
    g_test = (paillier.n + 1) % paillier.n2  # 通常使用 g = n+1
    lu_test = L(pow(g_test, paillier.lambda_dec, paillier.n2), paillier.n)
    print(f"g = {g_test}")
    print(f"L(g^λ mod N²) = {lu_test}")

    # 檢查模逆運算
    try:
        mu_test = pow(lu_test, -1, paillier.n)
        print(f"重新計算的μ = {mu_test}")
        print(f"原μ = {paillier.mu}")
        print(f"μ是否相等: {mu_test == paillier.mu}")
    except:
        print("μ計算失敗 - 可能lu_test與n不互質")


    # print("原始 Presence:", pres_vectors[0])
    # print("解密 Presence:", dec_pres[0])
    # print("原始 TF-IDF (整數):", tfidf_int[0])
    # print("解密 TF-IDF:", dec_tfidf[0])
    
    # # 6) 建立文件區塊
    # doc_blocks = []
    # for i, doc in enumerate(docs):
    #     block = {
    #         "doc_id": i + 1,
    #         "tokens": doc,
    #         "tf": tf_vectors[i],
    #         "presence": pres_vectors[i],
    #         "tfidf": tfidf_int[i],
    #         "enc_tf": [(c1, c2) for (c1, c2) in enc_tf[i]],
    #         "enc_presence": [(c1, c2) for (c1, c2) in enc_pres[i]],
    #         "enc_tfidf": [(c1, c2) for (c1, c2) in enc_tfidf[i]],
    #     }
    #     doc_blocks.append(block)

    # # 7) 建立加密索引樹
    # root = build_index_tree(paillier, doc_blocks)

    # # 8) 存成 JSON
    # with open("package/data.json", "w", encoding="utf-8") as f:
    #     json.dump({"blocks": doc_blocks}, f, ensure_ascii=False, indent=2)

    # print("[INFO] Data saved to data.json")
    # print("[INFO] Root node of encrypted index tree created")

    # # 9) 搜尋測試
    # query_keywords = [vocab[0]] if len(vocab) > 0 else ["kw2"]  # 使用實際詞彙
    # trapdoor = generate_trapdoor(paillier, vocab, query_keywords, h_sys)

    # top_docs = gbfs_search(paillier, root, trapdoor, top_k=5)
    # print("Top-k search results:", top_docs)


if __name__ == "__main__":
    demo()
import json
import math
from package.Homo.paillier import Paillier,Entity
from package.Dataprocess.dataprocessing import generate_corpus, compute_presence_vectors, compute_raw_tf_vectors,compute_tfidf_vectors
from package.Trapdoor.trapdoor import generate_trapdoor, gbfs_search
from package.Dataprocess.indextree import build_index_tree

def demo():
    # 1) 產生系統金鑰與 DO 公鑰
    paillier = Paillier.keygen(k=64)
    do_entity = Entity.gen_entity_key(paillier.n,paillier.g_core,paillier.n2)
    sys_entity = Entity.gen_entity_key(paillier.n,paillier.g_core,paillier.n2)  # 系統公鑰 pk_TA

    n_do, g_do, h_do = do_entity["pk"]
    n_sys, g_sys, h_sys = sys_entity["pk"]

    # 2) 模擬資料
    vocab, docs = generate_corpus(num_docs=5, vocab_size=5, avg_terms_per_doc=8, seed=42)
    print(f"[INFO] 產生的詞彙表: {vocab}")
    print(f"[INFO] 產生的文件: {docs}")

    # 3) 計算原始整數 TF 向量
    tf_vectors = compute_raw_tf_vectors(vocab, docs)
    pres_vectors = compute_presence_vectors(vocab, docs)
    
    print(f"\n[INFO] 原始整數 TF 向量: {tf_vectors}")
    print(f"[INFO] 出現向量: {pres_vectors}")
    
    # 使用修正後的 TF-IDF 計算
    tfidf_int = compute_tfidf_vectors(tf_vectors, pres_vectors)

    # 4) 加密 - 修正加密呼叫
    print(f"\n[INFO] 開始加密...")
    
    # 確保加密的資料格式正確
    enc_tf = []
    for doc_idx, doc in enumerate(tf_vectors):
        enc_doc = []
        for val in doc:
            if not isinstance(val, int) or val < 0:
                print(f"[ERROR] TF 值必須是非負整數，得到: {val} (類型: {type(val)})")
                continue
            encrypted_pair = paillier.encrypt(val, h_do)
            enc_doc.append(encrypted_pair)
        enc_tf.append(enc_doc)
    
    enc_pres = []
    for doc_idx, doc in enumerate(pres_vectors):
        enc_doc = []
        for val in doc:
            encrypted_pair = paillier.encrypt(val, h_sys)
            enc_doc.append(encrypted_pair)
        enc_pres.append(enc_doc)
    
    enc_tfidf = []
    for doc_idx, doc in enumerate(tfidf_int):
        enc_doc = []
        for val in doc:
            if not isinstance(val, int) or val < 0:
                print(f"[ERROR] TF-IDF 值必須是非負整數，得到: {val} (類型: {type(val)})")
                continue
            encrypted_pair = paillier.encrypt(val, h_do)
            enc_doc.append(encrypted_pair)
        enc_tfidf.append(enc_doc)

    print(f"[INFO] 加密完成！")

    # 5) 驗證解密
    print(f"\n[INFO] 驗證解密...")
    dec_tf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tf]
    dec_pres = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_pres]
    dec_tfidf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tfidf]

    # 比對原始
    # print("原始 TF:", tf_vectors[0])
    # print("解密 TF:", dec_tf[0])
    # print("原始 Presence:", pres_vectors[0])
    # print("解密 Presence:", dec_pres[0])
    print("原始 TF-IDF (整數):", tfidf_int[0])
    print("解密 TF-IDF:", dec_tfidf[0])
    
    # 6) 建立文件區塊
    doc_blocks = []
    for i, doc in enumerate(docs):
        block = {
            "doc_id": i + 1,
            "tokens": doc,
            "tf": tf_vectors[i],
            "presence": pres_vectors[i],
            "tfidf": tfidf_int[i],
            "enc_tf": [(c1, c2) for (c1, c2) in enc_tf[i]],
            "enc_presence": [(c1, c2) for (c1, c2) in enc_pres[i]],
            "enc_tfidf": [(c1, c2) for (c1, c2) in enc_tfidf[i]],
        }
        doc_blocks.append(block)

    # 7) 建立加密索引樹
    root = build_index_tree(paillier, doc_blocks)

    # 8) 存成 JSON
    with open("package/data.json", "w", encoding="utf-8") as f:
        json.dump({"blocks": doc_blocks}, f, ensure_ascii=False, indent=2)

    print("[INFO] Data saved to data.json")
    print("[INFO] Root node of encrypted index tree created")

    # 9) 搜尋測試
    vocab_list = ['kw1','kw2','kw3','kw4','kw5']
    vocab = input("輸入要搜尋的 keywords (逗號分隔): ")
    query_keywords = [kw.strip() for kw in vocab.split(",") if kw.strip()]
    print(f"\n[INFO] 查詢關鍵字: {query_keywords}")
    trapdoor = generate_trapdoor(paillier, vocab_list, query_keywords, h_sys)
    init_enc = paillier.encrypt(0, h_sys)  # 初始化為0的密文
    top_docs = gbfs_search(paillier, root, trapdoor,init_enc, top_k=5)
    print("Top-k search results:", top_docs)

    # def plaintext_match(raw_vector, query_vector):
    #     return sum(rv * qv for rv, qv in zip(raw_vector, query_vector))
    # vocab = input("輸入要搜尋的 keywords (逗號分隔): ")
    # query_keywords = [kw.strip() for kw in vocab.split(",") if kw.strip()]
    # query_vector = [1 if kw in query_keywords else 0 for kw in vocab_list]
    # print("[INFO] 查詢向量:", query_vector)
    # matches = []
    # for doc_id, tfidf_vec in enumerate(tfidf_int, start=1):
    #     score = plaintext_match(tfidf_vec, query_vector)
    #     matches.append((doc_id, score))
    # matches.sort(key=lambda x: x[1], reverse=True)
    # top_k = matches[:5]
    # print("Top-k search results:", top_k)



if __name__ == "__main__":
    demo()
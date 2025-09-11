import json
import math
from package.Homo.paillier import Paillier
from package.Dataprocess.dataprocessing import generate_corpus,compute_tf_vectors,compute_presence_vectors,compute_tfidf_vectors

def demo():
    # 1) 產生系統金鑰與 DO 公鑰
    paillier = Paillier.keygen(k=64)
    do_entity = paillier.gen_entity_key()
    sys_entity = paillier.gen_entity_key()  # 系統公鑰 pk_TA

    n_do, g_do, h_do = do_entity["pk"]
    n_sys, g_sys, h_sys = sys_entity["pk"]

    # 2) 模擬資料
    vocab, docs = generate_corpus(num_docs=5, vocab_size=5, avg_terms_per_doc=8, seed=42)

    # 3) 計算 TF 與 Presence 向量
    tf_vectors = compute_tf_vectors(vocab, docs)
    pres_vectors = compute_presence_vectors(vocab, docs)
    tfidf_float, tfidf_int = compute_tfidf_vectors(tf_vectors, pres_vectors)

    # 4) 加密
    enc_tf = [[paillier.encrypt(int(tf*100), h_do) for tf in doc] for doc in tf_vectors]
    enc_pres = [[paillier.encrypt(p, h_sys) for p in doc] for doc in pres_vectors]  # 使用系統公鑰
    enc_tfidf = [[paillier.encrypt(m, h_do) for m in doc] for doc in tfidf_int]

    # 5) 驗證解密
    dec_tf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tf]
    dec_pres = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_pres]
    dec_tfidf = [[paillier.strong_decrypt(c1) % paillier.n for (c1, c2) in doc] for doc in enc_tfidf]

    # 比對原始
    print("\nCompare originals (first 10 dims of doc1):")
    print(" TF orig    :", tf_vectors[0])
    print("Decrypted TF:", dec_tf[0])
    print(" PRES orig  :", pres_vectors[0])
    print("Decrypted Presence:", dec_pres[0])
    print(" TFIDF orig :", tfidf_int[0])
    print("Decrypted TF-IDF scaled:", dec_tfidf[0])
    # 6) 存成 JSON
    block = {
        "docs": docs,
        "tf": tf_vectors,
        "presence": pres_vectors,
        "tfidf": tfidf_int,
        "enc_tf": [[(c1, c2) for (c1, c2) in doc] for doc in enc_tf],
        "enc_presence": [[(c1, c2) for (c1, c2) in doc] for doc in enc_pres],
        "enc_tfidf": [[(c1, c2) for (c1, c2) in doc] for doc in enc_tfidf],
    }

    with open("package/Data/data.json", "w", encoding="utf-8") as f:
        json.dump({"blocks": [block]}, f, ensure_ascii=False, indent=2)

    print("[INFO] Data saved to data.json")

if __name__ == "__main__":
    demo()

# demo_dataprocess.py
import json
from package.Homo.paillier import Paillier
from package.Dataprocess.dataprocessing import (
    generate_corpus, compute_tf_vectors, compute_presence_vectors,
    compute_tfidf_vectors, encrypt_vector
)

def demo():
    # 1) 產生系統金鑰與一個實體 (用來加密)
    paillier = Paillier.keygen(k=256)
    ent = paillier.gen_entity_key()
    n, g_core, h_i = ent["pk"]

    # 2) 模擬資料
    vocab, docs = generate_corpus(num_docs=5, vocab_size=7, avg_terms_per_doc=8, seed=42)
    print("Vocabulary keyword:", len(vocab))
    print("Docs keywords:")
    for i, d in enumerate(docs, start=1):
        print(f" doc{i}:", d)

    # 3) TF vectors (counts)
    tf_vectors = compute_tf_vectors(vocab, docs)
    print("\nTF vectors (counts):")
    for i, v in enumerate(tf_vectors, start=1):
        print(f" doc{i}:", v)

    # 4) Presence vectors (0/1)
    pres_vectors = compute_presence_vectors(vocab, docs)
    print("\nPresence vectors (0/1):")
    for i, v in enumerate(pres_vectors, start=1):
        print(f" doc{i}:", v)

    # 5) TF-IDF (float + int scaled)
    tfidf_float, tfidf_int = compute_tfidf_vectors(vocab, docs, scale=100)
    print("\nTF-IDF (float) sample:")
    print(tfidf_float[0][:10])
    print("TF-IDF (scaled int) first doc:")
    print(tfidf_int[0][:10])

    # 6) 加密示例（以第一個文件為例）
    print("\nEncrypting vectors for doc1 (first doc) ...")
    enc_tf = [paillier.encrypt(m, h_i) for m in tf_vectors[0]]
    enc_pres = [paillier.encrypt(m, h_i) for m in pres_vectors[0]]
    enc_tfidf = [paillier.encrypt(m, h_i) for m in tfidf_int[0]]


    # 7) 驗證：直接用 strong_decrypt 還原
    dec_tf = [paillier.strong_decrypt(c1) % paillier.n for (c1, _) in enc_tf]
    dec_pres = [paillier.strong_decrypt(c1) % paillier.n for (c1, _) in enc_pres]
    dec_tfidf = [paillier.strong_decrypt(c1) % paillier.n for (c1, _) in enc_tfidf]


    # 比對原始
    print("\nCompare originals (first 10 dims of doc1):")
    print(" TF orig    :", tf_vectors[0][:10])
    print("Decrypted TF:", dec_tf)
    print(" PRES orig  :", pres_vectors[0][:10])
    print("Decrypted Presence:", dec_pres)
    print(" TFIDF orig :", tfidf_int[0][:10])
    print("Decrypted TF-IDF scaled:", dec_tfidf)
    
    # 8) 將結果存成 JSON
    block = {
        "doc_id": 1,
        "tf": tf_vectors[0],
        "presence": pres_vectors[0],
        "tfidf": tfidf_int[0],
        "enc_tf": [[c1, c2] for (c1, c2) in enc_tf],
        "enc_presence": [[c1, c2] for (c1, c2) in enc_pres],
        "enc_tfidf": [[c1, c2] for (c1, c2) in enc_tfidf],
        "dec_tf": dec_tf,
        "dec_presence": dec_pres,
        "dec_tfidf": dec_tfidf
    }

    chain_data = {"blocks": [block]}

    with open("package\\Data\\data.json", "w", encoding="utf-8") as f:
        json.dump(chain_data, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Data saved to data.json")

if __name__ == "__main__":
    demo()

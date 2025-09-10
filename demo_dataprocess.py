# demo_dataprocess.py
from package.Homo.paillier import Paillier
from package.Dataprocess.dataprocessing import (
    generate_corpus, compute_tf_vectors, compute_presence_vectors,
    compute_tfidf_vectors, encrypt_vector, decrypt_vector_with_strong
)

def demo():
    # 1) 產生系統金鑰與一個實體 (用來加密)
    paillier = Paillier.keygen(k=256)
    ent = paillier.gen_entity_key()
    n, g_core, h_i = ent["pk"]

    # 2) 模擬資料
    vocab, docs = generate_corpus(num_docs=5, vocab_size=15, avg_terms_per_doc=8, seed=42)
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
    print("\nTF-IDF (float) sample (first doc first 10 dims):")
    print(tfidf_float[0][:10])
    print("TF-IDF (scaled int) first doc (first 10 dims):")
    print(tfidf_int[0][:10])

    # 6) 加密示例（以第一個文件為例）
    print("\nEncrypting vectors for doc1 (first doc) ...")
    enc_tf = encrypt_vector(paillier, h_i, tf_vectors[0])
    enc_pres = encrypt_vector(paillier, h_i, pres_vectors[0])
    enc_tfidf = encrypt_vector(paillier, h_i, tfidf_int[0])

    # 7) 驗證：用強私鑰解密來還原明文（demo only）
    dec_tf = decrypt_vector_with_strong(paillier, enc_tf)
    dec_pres = decrypt_vector_with_strong(paillier, enc_pres)
    dec_tfidf = decrypt_vector_with_strong(paillier, enc_tfidf)

    print("\nDecrypted TF (doc1):", dec_tf)
    print("Decrypted Presence (doc1):", dec_pres)
    print("Decrypted TF-IDF scaled (doc1):", dec_tfidf)

    # 比對原始
    print("\nCompare originals (first 10 dims):")
    print(" TF orig    :", tf_vectors[0][:10])
    print(" PRES orig  :", pres_vectors[0][:10])
    print(" TFIDF orig :", tfidf_int[0][:10])
    print(" TFIDF dec  :", dec_tfidf[:10])

if __name__ == "__main__":
    demo()

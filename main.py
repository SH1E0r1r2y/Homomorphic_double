# main.py
from package import Paillier

if __name__ == "__main__":
    paillier = Paillier.keygen(K=64) # 64-bit 大小
    print("[*] System keys")
    print(f"N bits ~= {paillier.N.bit_length()}")
    #print(f"g_core   = {paillier.g_core}")
    print(f"g_enc    = {paillier.g_enc}   # should be N+1")
    print(f"lambda_dec = {paillier.lambda_dec}")
    #print(f"mu       = {paillier.mu}")
    #print(f"repr     = {paillier!r}")
    print(f"mu      = {paillier.mu}")

    # 產出節點的 (pk, sk_weak)
    ent = paillier.gen_entity_key(paillier.N, paillier.N2, paillier.g_core)
    print(f"pk_i    = (N, g_core, h_i) = {ent['pk']}")
    print(f"theta_i = {ent['sk_weak']}")
    print("\n[*] Encrypt/Decrypt demo with g_enc (N+1)")
    m = 15005467 % paillier.N
    c1,c2 = Paillier.encrypt(m, paillier.N, paillier.N2,paillier.g_enc,)
    m_dec = Paillier.strong_decrypt(c1, paillier.N, paillier.N2, paillier.lam_dec, paillier.mu)
    print(f"m  = {m}")
    print(f"c  = {c1}")
    print(f"m' = {m_dec}")
    assert m == m_dec, "解密錯誤：m != m'"
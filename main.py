# main.py
from package.Homo.paillier import system_keygen, gen_entity_key, encrypt_int, decrypt_int

def key_gen_paillierTA(bit_length=64):
    syskeys = system_keygen(bit_length=bit_length)
    N, N2 = syskeys["N"], syskeys["N2"]
    g_enc  = syskeys["g_enc"]    # Paillier 用
    g_core = syskeys["g_core"]   # 子群用
    lam_dec = syskeys["lambda_dec"]
    mu = syskeys["mu"]

    print("[*] System keys")
    print(f"N bits ~= {N.bit_length()}")
    print(f"g_core  = {g_core}")
    print(f"g_enc   = {g_enc}   # should be N+1")
    print(f"lambda_dec  = {lam_dec}")
    print(f"mu      = {mu}")

    # 產出節點的 (pk, sk_weak)
    ent = gen_entity_key(N, N2, g_core)
    print(f"pk_i    = (N, g_core, h_i) = {ent['pk']}")
    print(f"theta_i = {ent['sk_weak']}")
    print("\n[*] Encrypt/Decrypt demo with g_enc (N+1)")
    # 簡單測試加解密
    m = 15005467 % N
    c = encrypt_int(m, N, N2, g_enc)   # 這裡用 g_enc
    m_dec = decrypt_int(c, N, N2, lam_dec, mu)
    print(f"m  = {m}")
    print(f"c  = {c}")
    print(f"m' = {m_dec}")
    assert m == m_dec, "解密錯誤：m != m'"
if __name__ == "__main__":
    key_gen_paillierTA(bit_length=64) # 示範用 64-bit

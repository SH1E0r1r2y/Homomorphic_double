# main.py
from package.Homo.paillier import Paillier

def main():
    # 1) 產生系統金鑰（對齊你的輸出格式）
    paillier = Paillier.keygen(K=64)
    print("[*] System keys")
    print("N bits ~= ", paillier.N.bit_length())
    print("g_enc    =", paillier.g_enc)        # 應該等於 N + 1
    print("lambda_dec =", paillier.lambda_dec)
    print("mu      =", paillier.mu)

    # 2) 產生一個實體金鑰（pk_i 與 θ_i）
    ent = paillier.gen_entity_key()
    N, g_core, h_i = ent["pk"]
    theta_i = ent["sk_weak"]

    # 3) 加密 → 解密
    m = 15005467 % N
    C1, C2 = paillier.encrypt(m, h=h_i)
    m_strong = paillier.strong_decrypt(C1)
    m_weak = paillier.weak_decrypt(C1, C2, theta_i)
    print("[*] Encrypt/Decrypt demo")
    print("m      =", m)
    print("m_strong  =", m_strong, "(強私鑰)")
    print("m_weak    =", m_weak,   "(弱私鑰)")
    assert m_strong == m
    assert m_weak   == m
    print("[OK] 解密結果一致。")

    # 4) 重新隨機化（不改變明文）
    C1r, C2r = paillier.ciphertext_refresh((C1, C2), h=h_i)
    m_r = paillier.strong_decrypt(C1r)
    print("[*] Refresh ok? ", m_r == m)

    # 5) 同態加法
    m2 = 9876 % N
    D1, D2 = paillier.encrypt(m2, h=h_i)
    S1, S2 = Paillier.homomorphic_add((C1, C2), (D1, D2), paillier.N2)
    sum_dec = paillier.strong_decrypt(S1)
    print("[*] Homomorphic add ok? ", sum_dec == ((m + m2) % N))

if __name__ == "__main__":
    main()

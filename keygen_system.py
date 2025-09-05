# main.py
from package.Homo.paillier import Paillier
from package.Homo.promise import PedersenVSS

def paillier_demo():
    """
    1) 產生系統金鑰（對齊你的輸出格式）
    2) 使用系統公鑰 h_TA 進行加密
    3) 用 系統強私鑰(λ) 與 弱私鑰(theta_ta) 解密驗證
    """
    paillier = Paillier.keygen(k=64)
    print("[*] System keys")
    print("N bits ~= ", paillier.n.bit_length())
    print("lambda_dec =", paillier.lambda_dec)
    print("mu      =", paillier.mu)

    ent = paillier.gen_entity_key()
    n, g_core, h_i = ent["pk"]
    theta_i = ent["sk_weak"]

    m = 15005468827 % n
    c1, c2 = paillier.encrypt(m, h=h_i)
    m_strong = paillier.strong_decrypt(c1)
    m_weak = paillier.weak_decrypt(c1, c2, theta_i)
    print("[*] Encrypt/Decrypt demo")
    print("m      =", m)
    print("m_strong  =", m_strong, "(強私鑰)")
    print("m_weak    =", m_weak,   "(弱私鑰)")
    assert m_strong == m
    assert m_weak   == m
    print("[OK] 解密結果一致。")

    c1r, c2r = paillier.ciphertext_refresh((c1, c2), h=h_i)
    m_r = paillier.strong_decrypt(c1r)
    print("[*] Refresh ok? ", m_r == m)

    # 5) 同態加法
    m2 = 9876 % n
    d1, d2 = paillier.encrypt(m2, h=h_i)
    s1, s2 = Paillier.homomorphic_add((c1, c2), (d1, d2), paillier.n2)
    sum_dec = paillier.strong_decrypt(s1)
    print("[*] Homomorphic add ok? ", sum_dec == ((m + m2) % n))

def vss_demo():
    """
    1) 產生 (p, q, α, β)
    2) 將某個秘密 s 做 3-out-of-5 分享
    """
    vss = PedersenVSS.keygen(min_q_bits=256)

    s = 123456789 % vss.q
    t, d = 3, 5
    e0, es, shares = vss.init(s, t, d)

    # 3) 每份 share 本地驗證
    for (i, si, vi) in shares:
        assert vss.verify(i, si, vi, e0, es, t), f"Share {i} 驗證失敗"

    # 4) 任取 t 份重建秘密
    subset = shares[:t]
    s_rec = vss.recover([(i, si) for (i, si, _vi) in subset],t)
    print("[VSS] s original =", s)
    print("[VSS] s recover  =", s_rec)
    assert s_rec == s


if __name__ == "__main__":
    paillier_demo()
    vss_demo()

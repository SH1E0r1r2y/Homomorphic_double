# main.py
from package.Homo.paillier import Paillier
from package.Homo.promise import PedersenVSS
from package.Homo.indextree import IndexTreeNode, build_index_tree


def paillier_commit(t: int = 3, d: int = 5):
    # 1) 產生 Paillier 系統鑰（含 λ 與 θ_TA）
    """
    1) 產生系統金鑰（對齊你的輸出格式）
    2) 使用系統公鑰 h_TA 進行加密
    3) 用 系統強私鑰(λ) 與 弱私鑰(theta_ta) 解密驗證
    """
    paillier = Paillier.keygen(k=256)
    print("[*] System keys")
    print("N bits ~= ", paillier.n.bit_length())

    ent = paillier.gen_entity_key()
    n, g_core, h_i = ent["pk"]
    theta_i = ent["sk_weak"]

    m = 15005468827 % n
    c1, c2 = paillier.encrypt(m, paillier.h_ta)
    m_strong = paillier.strong_decrypt(c1)
    m_weak = paillier.weak_decrypt(c1, c2, paillier.theta_ta)
    print("[*] Encrypt/Decrypt demo")
    print("m      =", m)
    print("m_strong  =", m_strong, "系統(強私鑰)")
    print("m_weak    =", m_weak,   "系統(弱私鑰)")
    assert m_strong == m
    assert m_weak   == m
    print("[OK] 解密結果一致。")

    """
    依定義對系統強/弱私鑰做 Pedersen 承諾與 t-out-of-d 秘密分享：
      E0_λ = α^λ β^v,  E0_θ = α^θ β^{v'}
      shares: (i, λ_i, v_i), (i, θ_i, v'_i)
    """
    # 2) 產生 VSS 參數 (p, q, α, β) —— 同一組參數同時承諾/分享 λ 與 θ_TA
    vss = PedersenVSS.keygen(min_q_bits=256)

    # 3) 依群階 q 取值（Pedersen 承諾的秘密皆在 Z_q 上）
    s_lambda = paillier.lambda_dec % vss.q
    s_theta  = paillier.theta_ta  % vss.q

    # 4) 對 λ 與 θ_TA 各自建立承諾與 t-of-d 分享（f,g 多項式內部由 VSS 實作）
    e0_lambda, es_lambda, shares_lambda = vss.init(s_lambda, t, d)  # E(λ, v)
    e0_theta,  es_theta,  shares_theta  = vss.init(s_theta,  t, d)  # E(θ, v')

    # 5) 本地驗證每一份 share
    for (i, si, vi) in shares_lambda:
        assert vss.verify(i, si, vi, e0_lambda, es_lambda, t), f"[λ] Share {i} 驗證失敗"
    for (i, si, vi) in shares_theta:
        assert vss.verify(i, si, vi, e0_theta, es_theta, t),   f"[θ] Share {i} 驗證失敗"

    # 6) 將 shares 依節點 i 配對（分配給節點 A_i）
    by_id = {i: {} for i in range(1, d + 1)}
    for (i, si, vi) in shares_lambda:
        by_id[i]["lambda_i"] = si
        by_id[i]["v_i"]      = vi
    for (i, si, vi) in shares_theta:
        by_id[i]["theta_i"]      = si
        by_id[i]["v_i_prime"]    = vi

    # 7) 抽 t 份重建自我檢查（模 q）
    idxs = list(range(1, t + 1))
    lam_rec = vss.recover([(i, by_id[i]["lambda_i"]) for i in idxs], t)
    th_rec  = vss.recover([(i, by_id[i]["theta_i"])  for i in idxs], t)

    print("\n[*] VSS 重建檢查")
    print("原始 λ =", s_lambda, " | 重建 λ =", lam_rec)
    print("原始 θ =", s_theta,  " | 重建 θ =", th_rec)
    assert lam_rec == s_lambda, "λ 重建不相等"
    assert th_rec  == s_theta,  "θ 重建不相等"

    # 聚合不同節點的 shares
    idxs = list(by_id.keys())[:t]
    shares_lambda = [(i, by_id[i]['lambda_i']) for i in idxs]
    shares_theta  = [(i, by_id[i]['theta_i'])  for i in idxs]

    # 重建秘密
    lam_rec = vss.recover(shares_lambda, t)
    th_rec  = vss.recover(shares_theta, t)


    assert lam_rec == s_lambda, "λ重建不符"
    assert th_rec  == s_theta, "θ重建不符"
    print(f"[Test] 節點協作重建密鑰成功 (t={t})")

    # 8) 輸出摘要
    print("\n[*] Pedersen 承諾（系統強/弱私鑰）")
    print("E0_lambda =", e0_lambda)
    print("E0_theta  =", e0_theta)
    print("[*] 係數承諾 Es_lambda  =", len(es_lambda), ", Es_theta 長度 =", len(es_theta))
    print(f"[*] 已建立 {d} 份 shares，門檻 t = {t}")
    for i in range(1, d + 1):
        rec = by_id[i]
        print(f" A_{i}: (λ_i={rec['lambda_i']} || (θ_i={rec['theta_i']}")

    return {
        "paillier": paillier,
        "vss": vss,
        "E0_lambda": e0_lambda, "Es_lambda": es_lambda, "shares_lambda": shares_lambda,
        "E0_theta":  e0_theta,  "Es_theta":  es_theta,  "shares_theta":  shares_theta,
        "node_shares": by_id
    }

def index_tree_demo():
    # 1. 系統初始化（Paillier 公私鑰）
    paillier = Paillier.keygen(k=64)
    ent = paillier.gen_entity_key()
    n, g_core, h_i = ent["pk"]

    # 2. 模擬文件索引向量 (明文) —— 每個文件用長度=3 的向量表示
    doc_vectors = [
        [1, 0, 2],  # 文件1
        [0, 1, 1],  # 文件2
        [1, 1, 0],  # 文件3
        [2, 0, 1],  # 文件4
    ]

    # 3. 加密向量
    leaf_nodes = []
    for doc_id, vec in enumerate(doc_vectors, start=1):
        enc_vec = [paillier.encrypt(val, h=h_i) for val in vec]
        leaf = IndexTreeNode(
            node_id=f"doc-{doc_id}",
            index_vector=enc_vec,
            fid=doc_id
        )
        leaf_nodes.append(leaf)

    # 4. 建立 index tree
    root = build_index_tree(leaf_nodes, paillier.n2)

    # 5. 輸出樹結構
    def print_tree(node, depth=0):
        indent = "  " * depth
        if node.fid:
            print(f"{indent}- Leaf Node {node.id}, fid={node.fid}")
        else:
            print(f"{indent}- Internal Node {node.id}")
        if node.Pl:
            print_tree(node.Pl, depth + 1)
        if node.Pr:
            print_tree(node.Pr, depth + 1)

    print("[*] Index Tree 結構:")
    print_tree(root)

if __name__ == "__main__":
    index_tree_demo()
    paillier_commit(t=3, d=5)

import time
import json
import math
import heapq
import itertools
from package.Homo.paillier import Paillier, Entity, FunctionNode
from package.Homo.promise import PedersenVSS
from package.Dataprocess.dataprocessing import (
compute_local_cid,generate_corpus,compute_presence_vectors,compute_raw_tf_vectors,compute_tfidf_vectors
)
from package.Trapdoor.trapdoor import generate_trapdoor, gbfs_match_sort,score_and_rank
from package.Dataprocess.indextree import build_index_tree

def print_stage(title: str):
    print("\n" + "="*10 + f" {title} " + "="*10)
def log(role: str, msg: str, duration: float = None):
    if duration is None:
        print(f"[{role}] {msg}")
    else:
        print(f"[{role}] {msg} 耗時: {duration:.4f} 秒")

def setup_system(t: int, d: int, k: int):
    timings = {}
    start = time.time()

    # 生成 Paillier 系統金鑰
    paillier = Paillier.keygen(k)
    timings['keygen'] = time.time() - start
    log("TA", f"產生系統金鑰，N 長度 {k} bits", timings['keygen'])

    # 初始化 Pedersen VSS 系統
    start = time.time()
    vss = PedersenVSS.keygen(k*2)  # q 約為 N^2 的大小
    # print(f"VSS 參數: q={vss.q}")
    # print(f"Paillier 參數: n={paillier.n}, n2={paillier.n2}, g_core={paillier.g_core}, λ={paillier.lambda_dec}, θ={paillier.theta_ta}, h_ta={paillier.h_ta}, μ={paillier.mu}")

    # 將 Paillier 的秘密映射到 Z_q
    s_lambda = paillier.lambda_dec % vss.q
    s_theta  = paillier.theta_ta % vss.q

    # 生成秘密分片與承諾
    e0_l, es_l, shares_l = vss.init(s_lambda, t, d)  # λ 的分片
    e0_t, es_t, shares_t = vss.init(s_theta, t, d)   # θ 的分片
    timings['vss'] = time.time() - start
    log("TA", f"生成 Pedersen VSS 並分發 {d} 份秘密分享 (t={t})", timings['vss'])

    # 建立 FunctionNode，每個節點保存 share 與 Pedersen 承諾
    fns = {}
    for i in range(1, d+1):
        s_i, v_i = shares_l[i-1][1], shares_l[i-1][2]  # λ_share 分片與 Pedersen 隨機數
        lam_i = s_i
        the_i = shares_t[i-1][1]  # θ_share
        pedersen_commit_i = vss.commit(s_i, v_i)  # 生成承諾
        fns[i] = FunctionNode(
            id=i,
            paillier=paillier,
            lambda_share=lam_i,
            theta_share=the_i,
            pedersen_commit=pedersen_commit_i,
            v_i=shares_l[i-1][2],
            v_i_prime=shares_t[i-1][2]
        )
        log(f"FN{i}", "接收 λ, θ 分片並生成 Pedersen 承諾")

    timings['total'] = sum(timings.values())
    log("TA", "setup_system 完成", timings['total'])

    return paillier, vss, fns, e0_l, es_l, e0_t, es_t

def simulate_do_data_upload(do: Entity, paillier: Paillier):
    timings = {}
    start = time.time()
    vocab, docs = generate_corpus(num_docs=7, vocab_size=5, avg_terms_per_doc=15, seed=42)
    log(f"DO {do.id}", "產生 7 筆模擬文件，每筆 15 個關鍵字")
    log(f"DO {do.id}", f"驗證文件內容: {docs}")

    cid_map = {}
    for i, tokens in enumerate(docs, start=1):
        block = {"doc_id": i, "tokens": tokens}
        data = json.dumps(block).encode("utf-8")
        cid_map[i] = compute_local_cid(data)

    tf = compute_raw_tf_vectors(vocab, docs)
    log(f"DO {do.id}", f"原始整數 TF 向量: {tf}")
    pres = compute_presence_vectors(vocab, docs)
    #tfidf_i = compute_tfidf_vectors(tf, pres)
    log(f"DO {do.id}", "計算 TF/Presence")
    timings['vector'] = time.time() - start
    start = time.time()
    n, g, h_do = do.pk
    enc_tf = [[paillier.encrypt(v, h_do) for v in vec] for vec in tf]
    #enc_tfidf = [[paillier.encrypt(v, h_do) for v in vec] for vec in tfidf_i]
    enc_pres = [[paillier.encrypt(v, paillier.h_ta) for v in vec] for vec in pres]
    timings['encrypt'] = time.time() - start
    log(f"DO {do.id}", "加密 TF, 系統加密 Presence")
    timings['total'] = timings['vector'] + timings['encrypt']
    log(f"DO {do.id}", "simulate_do_data_upload 完成", timings['total'])
    return vocab, enc_tf, enc_pres, cid_map

def build_blockchain_data(do: Entity, paillier: Paillier, cid_map, vocab, enc_tf, enc_pres):
    start = time.time()
    blocks = []
    EI_prime_list = []

    print("=== 建立區塊鏈資料與葉節點 ===")
    for i in range(len(enc_tf)):
        block = {
            "CID": cid_map[i+1],
            "doc_id": i+1,
            "owner_id": do.id,
            "vocab": vocab,
            "enc_tf": enc_tf[i],
            "enc_presence": enc_pres[i],
        }
        blocks.append(block)
        EI_prime_list.append(enc_pres[i])
        #print(f"Leaf {i+1}: doc_id={block['doc_id']}, CID={block['CID']}, enc_presence={block['enc_presence']}")

    root = build_index_tree(paillier, blocks)
    duration = time.time() - start
    log("Blockchain", "建立索引樹", duration)
    print(f"=== 完成索引樹建立, root 節點 enc_vector: {root.enc_vector} ===")
    #print(f"{EI_prime_list}")
    return root, EI_prime_list

def simulate_du_query(du, paillier, vocab, root, fns, t, vss, e0_l, es_l, e0_t, es_t, EI_prime_list, top_k=5):
    counter = itertools.count()
    raw = input("[DU Bob] 輸入要搜尋的 keywords (逗號分隔): ")
    query_keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]

    # Step 1: DU 用弱私鑰加密生成陷門 ET
    warrant, ET, top_k = generate_trapdoor(paillier, vocab, query_keywords, du, du.sk_weak, top_k)

    # Step 2: FN 協作閾值解密 ET → 得到明文 T
    participants = [fns[i] for i in range(1, t+1)]
    partials = []
    for fn in participants:
        if fn.verify_share(vss, e0_l, es_l, e0_t, es_t, t):
            pv = [fn.partial_decrypt(ct) for ct in ET]
            partials.append((fn.id, pv))

    lambda_shares = [(fn.id, fn.lambda_share) for fn in participants]
    lambda_recov = vss.recover(lambda_shares, t)
    seta_shares = [(fn.id, fn.theta_share) for fn in participants]
    seta_recov  = vss.recover(seta_shares, t)

    T_plain = [paillier.strong_decrypt(c, lambda_recov) for c in ET]

    print(f"\n[Debug] 重建 λ = {lambda_recov}")
    print(f"[Debug] λ 原始: {paillier.lambda_dec}")
    print(f"[Debug] 重建 θ = {seta_recov}")
    print(f"[Debug] θ 原始: {paillier.theta_ta}")
    print("[Debug] T_plain (解密後的陷門向量):")
    print(T_plain)

    candidate_nodes = gbfs_match_sort(
    paillier, root, T_plain, participants, seta_recov, lambda_recov, top_k
)

    # Step B: 執行 Score Sort 和最終排名 (第二階段)
    final_ranked_results = score_and_rank(
        paillier, root, T_plain, lambda_recov, top_k, candidate_nodes
    )

    log(f"DU {du.id}", f"Top-{top_k} 搜尋結果: {final_ranked_results}")
    return final_ranked_results

if __name__ == "__main__":
    print_stage("系統啟動")
    paillier, vss, fns, e0_l, es_l, e0_t, es_t = setup_system(t=3, d=5, k=64)
    print_stage("註冊 Data Owner 並上傳資料")
    do = Entity.register_data_owner(paillier, "Alice")
    vocab, enc_tf, enc_pres, cid_map = simulate_do_data_upload(do, paillier)
    print_stage("上傳區塊鏈，建立 Index Tree")
    root, EI_prime_list = build_blockchain_data(do, paillier, cid_map, vocab, enc_tf, enc_pres)
    print_stage("註冊 Data User 並查詢")
    du = Entity.register_data_user(paillier, "Bob")
    results = simulate_du_query(du, paillier, vocab, root, fns, t=3,vss=vss, 
                                e0_l=e0_l, es_l=es_l, e0_t=e0_t, es_t=es_t, EI_prime_list=EI_prime_list)
    print_stage("搜尋結果")
    print(f"Top-5 文件 (ID, 匹配度)：{results}")
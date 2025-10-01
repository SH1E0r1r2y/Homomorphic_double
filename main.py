import time
import json
from package.Homo.paillier import Paillier, Entity, FunctionNode
from package.Homo.promise import PedersenVSS
from package.Dataprocess.dataprocessing import (
compute_local_cid,generate_corpus,compute_presence_vectors,compute_raw_tf_vectors,compute_tfidf_vectors
)
from package.Trapdoor.trapdoor import generate_trapdoor, gbfs_search
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
    paillier = Paillier.keygen(k)
    timings['keygen'] = time.time() - start
    log("TA", f"產生系統金鑰，N 長度 {k} bits", timings['keygen'])
    start = time.time()
    vss = PedersenVSS.keygen(k*2)
    s_lambda = paillier.lambda_dec % vss.q
    s_theta = paillier.theta_ta  % vss.q
    print(f"paillier.lambda_dec: {paillier.lambda_dec}, paillier.theta_ta: {paillier.theta_ta}")
    print(f"TA's secret λ: {paillier.lambda_dec}, θ: {paillier.theta_ta}")
    e0_l, es_l, shares_l = vss.init(s_lambda, t, d)
    e0_t, es_t, shares_t = vss.init(s_theta, t, d)
    timings['vss'] = time.time() - start
    log("TA", f"生成 Pedersen VSS 並分發 {d} 份秘密分享 (t={t})", timings['vss'])
    fns = {}
    for i in range(1, d+1):
        lam_i, v_i_l = shares_l[i-1][1:]
        the_i, v_i_t = shares_t[i-1][1:]

        fns[i] = FunctionNode(
            id=i,
            paillier=paillier,
            lambda_share=lam_i,
            theta_share=the_i,
            pedersen_commit=(e0_l, es_l, e0_t, es_t),  # or 分別給 λ, θ 的 commit
            v_i=(v_i_l, v_i_t)
        )
        log(f"FN{i}", "接收 λ, θ 分片")
    timings['total'] = sum(timings.values())
    log("TA", "setup_system 完成", timings['total'])
    return  paillier, vss, fns, e0_l, es_l, e0_t, es_t

def simulate_do_data_upload(do: Entity, paillier: Paillier):
    timings = {}
    start = time.time()
    vocab, docs = generate_corpus(num_docs=5, vocab_size=5, avg_terms_per_doc=15, seed=42)
    log(f"DO {do.id}", "產生 5 筆模擬文件，共 kw1~kw5 個關鍵字")
    log(f"DO {do.id}", f"驗證文件內容: {docs}")
    cid_map = {}
    for i, tokens in enumerate(docs, start=1):
        block = {"doc_id": i, "tokens": tokens}
        data = json.dumps(block).encode("utf-8")
        cid_map[i] = compute_local_cid(data)
    tf = compute_raw_tf_vectors(vocab, docs)
    pres = compute_presence_vectors(vocab, docs)
    log(f"DO {do.id}", f"整數 TF 向量: {tf}")
    log(f"DO {do.id}", f"原始 Presence 向量: {pres}")
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

def build_blockchain_data(do: Entity, paillier: Paillier, cid_map,vocab, enc_tf, enc_pres):
    start = time.time()
    blocks = []
    for i in range(len(enc_tf)):
        blocks.append({
            "CID": cid_map[i+1],
            "doc_id": i+1,
            "owner_id": do.id,
            "vocab": vocab,
            "enc_tf": enc_tf[i],
            "enc_presence": enc_pres[i],
            #"enc_tfidf": enc_tfidf[i]
        })
    root = build_index_tree(paillier, blocks)
    duration = time.time() - start
    log("Blockchain", "建立索引樹", duration)
    return root

def simulate_du_query(du: Entity, paillier: Paillier, vocab, root, fns, t: int, vss=None,
                      e0_l=None, es_l=None, e0_t=None, es_t=None):
    import time
    start = time.time()
    raw = input("[DU Bob] 輸入要搜尋的 keywords (逗號分隔): ")
    query_keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
    sk_du = du.sk_weak

    # Step 1. 生成 Trapdoor
    ET = generate_trapdoor(paillier, vocab, query_keywords, sk_du)
    log(f"DU {du.id}", "生成 Trapdoor", time.time() - start)

    # Step 2. 功能節點部分解密
    start = time.time()
    participants = list(fns.values())[:t]
    partials = []
    for fn in participants:
        if fn.verify_share(vss, e0_l, es_l, e0_t, es_t, t):
            pv = [fn.partial_decrypt(ct) for ct in ET]
            partials.append((fn.id, pv))

    # Step 3. VSS 重建 λ, θ
    lambda_shares = [(fn.id, fn.lambda_share) for fn in participants]
    lambda_recov = vss.recover(lambda_shares, t)
    seta_shares = [(fn.id, fn.theta_share) for fn in participants]
    seta_recov = vss.recover(seta_shares, t)

    # Step 4. 強解密 Trapdoor 向量 (得到 T_plain，通常是 0/1 向量)
    T_plain = [paillier.strong_decrypt(ciphertext, lambda_recov) for ciphertext in ET]
    
    print("[Debug] T_plain (解密後的陷門向量):")
    print(T_plain)

    log(f"DU {du.id}", "閾值解密 & 強解密 Trapdoor", time.time() - start)

    # Step 5. 保留原始維度 (m)，只在 T[j]==1 的位置放 ET，其餘用 (1,1)
    start = time.time()
    simplified_ET = [
        ET[idx] if val == 1 else (1, 1)
        for idx, val in enumerate(T_plain)
    ]

    print(f"[Debug] simplified ET (只保留 T[j]==1 的位置): {simplified_ET}")
    log(f"DU {du.id}", f"生成 simplified ET (count={len(simplified_ET)})", time.time() - start)

    # Step 6. 搜尋
    start = time.time()
    init_enc = paillier.encrypt(0, paillier.h_ta)
    results = gbfs_search(paillier, root, simplified_ET, seta_recov, top_k=5, t_plain=T_plain)
    log(f"DU {du.id}", "執行搜尋", time.time() - start)
    # 輸出結果
    print_stage("搜尋結果")
    for score, cid, doc_id in results:
        print(f"匹配度: {score}, CID: {cid}, doc_id: {doc_id}")
    return results

if __name__ == "__main__":
    print_stage("系統啟動")
    paillier, vss, fns, e0_l, es_l, e0_t, es_t = setup_system(t=3, d=5, k=64)
    print_stage("註冊 Data Owner 並上傳資料")
    do = Entity.register_data_owner(paillier, "Alice")
    vocab, enc_tf, enc_pres, cid_map = simulate_do_data_upload(do, paillier)
    print_stage("上傳區塊鏈，建立 Index Tree")
    root = build_blockchain_data(do, paillier, cid_map, vocab, enc_tf, enc_pres)
    print_stage("註冊 Data User 並查詢")
    du = Entity.register_data_user(paillier, "Bob")
    results = simulate_du_query(du, paillier, vocab, root, fns, t=3,
                            vss=vss, e0_l=e0_l, es_l=es_l, e0_t=e0_t, es_t=es_t)
    print_stage("搜尋結果")
    print(f"Top-5 文件 (ID, 匹配度)：{results}")
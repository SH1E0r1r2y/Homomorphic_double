# package/paillier.py
import random
from math import gcd, lcm
from sympy import randprime, isprime

def L(u: int, n: int) -> int:
    return (u - 1) // n

def _rand_coprime(modulus: int) -> int:
    while True: #取與 modulus 互質的整數，用來找隨機數 r ∈ Z*_{N}
        x = random.randrange(2, modulus - 1)
        if gcd(x, modulus) == 1:
            return x

def generate_strong_primes(bit_length: int):
    while True:
        p = randprime(2**(bit_length - 1), 2**bit_length)
        q = randprime(2**(bit_length - 1), 2**bit_length)
        if p != q and isprime((p - 1) // 2) and isprime((q - 1) // 2):
            return p, q

def find_g(p: int, q: int):
    N  = p * q
    N2 = N * N
    p_ = (p - 1) // 2
    q_ = (q - 1) // 2
    lam_g = lcm(p - 1, q - 1) // 2  # g 的階 = lcm(p', q')
    target = 2 * p_ * q_

    while True: # 直到挑到適合的 g
        while True: # 先找到與 N^2 互質的 a，隨機 a ∈ Z*_{N^2}，令 g ≡ -a^{2N} (mod N^2)
            a = random.randrange(2, N2 - 1)
            if gcd(a, N2) == 1:
                break
        g = (-pow(a, 2 * N, N2)) % N2
        if g == 1:
            continue  # trivial，重抽

        # A) gcd(L(g^λ mod N^2), N) == 1
        u = pow(g, lam_g, N2)
        Lu = L(u, N)
        if gcd(Lu, N) != 1:
            continue

        # B) 階度檢查：g^target = 1 且ord(g) = 2 * p' * q'（p'=(p-1)/2, q'=(q-1)/2）
        if pow(g, target, N2) != 1:
            continue

        bad = False
        for r in (2, p_, q_):
            if pow(g, target // r, N2) == 1:
                bad = True
                break
        if bad:
            continue

        return g, N, N2

def system_keygen(bit_length: int = 64):
    p, q = generate_strong_primes(bit_length)
    g_core, N, N2 = find_g(p, q)   # find_g 內部自己用的是 λ/2，OK
    g_enc = (N + 1) % N2              # Paillier 加解密用

    lambda_dec  = lcm(p - 1, q - 1)        # ← 完整 λ（解密用）
    lambda_core = lambda_dec // 2          # ← 半 λ（論文/子群用）

    # μ 用 g_enc 與「完整 λ」計算
    # μ = (L(g^λ mod N^2))^{-1} mod N
    Lu = L(pow(g_enc, lambda_dec, N2), N)
    mu = pow(Lu, -1, N)
    return {
        "p": p, "q": q,
        "N": N, "N2": N2,
        "g_core": g_core,
        "g_enc": g_enc,
        "lambda_dec": lambda_dec,
        "lambda_core": lambda_core,
        "mu": mu
    }

def gen_entity_key(N: int, N2: int, g_core: int):
    theta = random.randint(1, N // 4)
    h = pow(g_core, theta, N2) # θ_i ∈ [1, N/4],  h_i = g^{θ_i} mod N^2 
    return {"pk": (N, g_core, h), "sk_weak": theta}

def encrypt_int(m: int, N: int, N2: int, g_enc: int, r: int | None = None) -> int:
    if not (0 <= m < N):
        raise ValueError("message m 必須在 [0, N)")

    if r is None:
        r = _rand_coprime(N)  # 近似抽 Z*_N，用與 N 互質的亂數近似

    c = (pow(g_enc, m, N2) * pow(r, N, N2)) % N2 # c = g^m * r^N  (mod N^2)
    return c

def decrypt_int(c: int, N: int, N2: int, lam: int, mu: int) -> int:
    u = pow(c, lam, N2)
    m = (L(u, N) * mu) % N # 解密： m = L(c^λ mod N^2) * μ (mod N)
    return m

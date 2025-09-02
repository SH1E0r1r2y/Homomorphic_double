import random
from typing import Dict
from .utils import system_keygen,L,rand_coprime

class Paillier:
    def __init__(self, n:int, n2:int, g_enc:int,g_core:int ,lambda_dec:int, mu:int):
        self.N = n
        self.N2 = n2
        self.g_enc = g_enc
        self.g_core = g_core
        self.lambda_dec = lambda_dec
        self.mu = mu

    @classmethod
    def keygen(cls, K: int = 64) -> "Paillier":
        keys: Dict[str, int] = system_keygen(bit_length=K)
        return cls(
            n=keys["N"],
            n2=keys["N2"],
            g_enc=keys["g_enc"],
            g_core=keys["g_core"],
            lambda_dec=keys["lambda_dec"],
            mu=keys["mu"],
        )

    def gen_entity_key(N: int, N2: int, g_core: int):
        theta = random.randint(1, N // 4)
        h = pow(g_core, theta, N2) # θ_i ∈ [1, N/4],  h = g^{θ_i} mod N^2 
        return {"pk": (N, g_core, h), "sk_weak": theta}

    def encrypt(m: int, N: int, N2: int, g_enc: int, r: int | None = None, h: int | None = None) -> tuple[int, int]:
        if not (0 <= m < N):
            raise ValueError("message m 必須在 [0, N)")

        if r is None:
            r = rand_coprime(N)  # 近似抽 Z*_N，用與 N 互質的亂數近似

        c1 = ((1+ m * N) %N2) * pow(h, r, N2) % N2 #h_i = g^{θ_i} mod N^2
        c2 = pow(g_enc,r,N2)
        return c1,c2

    def strong_decrypt(c: int, N: int, N2: int, lam: int, mu: int) -> int:
        u = pow(c, lam, N2)
        m = (L(u, N) * mu) % N # 解密： m = L(c^λ mod N^2) * μ (mod N)
        return m

    def weak_decrypt(C1: int, C2: int, theta_i: int, N: int, N2: int) -> int:
        t = (C1 * pow(C2, theta_i, N2)) % N2
        return L(t, N)
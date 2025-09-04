# paillier.py
import random
from typing import Dict, Tuple
from .utils import system_keygen, L, rand_coprime

Cipher = Tuple[int, int]  # (C1, C2)

class Paillier:
    def __init__(self, n:int, n2:int, g_enc:int, g_core:int, lambda_dec:int, mu:int):
        self.N = n
        self.N2 = n2
        self.g_enc = g_enc       # 應為 N+1
        self.g_core = g_core     # 子群生成元（論文 g）
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
    # 產生一個實體的 (pk:(N, g_core, h), sk_weak:theta)
    def gen_entity_key(self, theta: int | None = None) -> Dict[str, Tuple[int, int, int]]:
        if theta is None:
            theta = random.randint(1, self.N // 4)
        h = pow(self.g_core, theta, self.N2)
        return {"pk": (self.N, self.g_core, h), "sk_weak": theta}

    # 加密：只需 m 與該實體的 h 
    def encrypt(self, m: int, h: int, r: int | None = None) -> Cipher:
        if not (0 <= m < self.N):
            raise ValueError("message m 必須在 [0, N)")
        if r is None:
            r = rand_coprime(self.N)  # 近似 Z*_N

        C1 = ((1 + m * self.N) % self.N2) * pow(h, r, self.N2) % self.N2
        C2 = pow(self.g_core, r, self.N2)
        return (C1, C2)

    # 強私鑰解密：論文公式 m = L(C1^λ mod N^2) * μ (mod N) 
    def strong_decrypt(self, C1: int) -> int:
        u = pow(C1, self.lambda_dec, self.N2)
        return (L(u, self.N) * self.mu) % self.N

    # 弱私鑰解密：剝除 h^r，需要乘上 (C2^theta) 的模逆
    def weak_decrypt(self, C1: int, C2: int, theta_i: int) -> int:
        inv = pow(pow(C2, theta_i, self.N2), -1, self.N2)
        val = (C1 * inv) % self.N2
        return L(val,self.N)

    # 重新隨機化（Ciphertext Refresh，不變明文）
    def ciphertext_refresh(self, C: Cipher, h: int, r_prime: int | None = None) -> Cipher:
        if r_prime is None:
            r_prime = rand_coprime(self.N)
        C1, C2 = C
        C1p = (C1 * pow(h, r_prime, self.N2)) % self.N2
        C2p = (C2 * pow(self.g_core, r_prime, self.N2)) % self.N2
        return (C1p, C2p)

    # 同態加法（兩個同一公鑰下的密文）
    @staticmethod
    def homomorphic_add(Ca: Cipher, Cb: Cipher, N2: int) -> Cipher:
        C1a, C2a = Ca
        C1b, C2b = Cb
        return ((C1a * C1b) % N2, (C2a * C2b) % N2)

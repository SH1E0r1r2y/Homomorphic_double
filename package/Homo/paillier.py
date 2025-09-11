# paillier.py
import random
from math import lcm
from typing import Dict, Tuple
from package.Homo.utils import modinv,L, rand_coprime, generate_strong_primes, find_g

Cipher = Tuple[int, int]  # (C1, C2)

class Paillier:
    """Paillier"""
    def __init__(self,n:int, n2:int, g_core:int, lambda_dec:int, theta_ta:int, h_ta:int, mu:int):
        self.n = n
        self.n2 = n2
        self.g_core = g_core     # 子群生成元g
        self.lambda_dec = lambda_dec
        self.theta_ta = theta_ta
        self.h_ta = h_ta
        self.mu = mu

    @classmethod
    def keygen(cls, k: int = 64) -> "Paillier":
        """產生金鑰"""
        p, q = generate_strong_primes(k)
        lambda_dec  = lcm(p - 1, q - 1)       # ← λ
        g_core, n, n2 = find_g(p, q, lambda_dec)
        # μ = (L(g^λ mod N^2))^{-1} mod N
        lu = L(pow((n + 1) % n2, lambda_dec, n2), n)
        mu = pow(lu, -1, n)
        theta_ta = random.randint(1, n // 4)
        h_ta = pow(g_core, theta_ta, n2)
        return cls(n=n, n2=n2, g_core=g_core, lambda_dec=lambda_dec, theta_ta=theta_ta, h_ta=h_ta, mu=mu)

    def gen_entity_key(self, theta: int | None = None) -> Dict[str, Tuple[int, int, int]]:
        """產生一個實體的 (pk:(N, g_core, h), sk_weak:theta)"""
        if theta is None:
            theta = random.randint(1, self.n // 4)
        h = pow(self.g_core, theta, self.n2)
        return {"pk": (self.n, self.g_core, h), "sk_weak": theta}

    def encrypt(self, m: int, h: int, r: int | None = None) -> Cipher:
        """加密：只需 m 與該實體的 h """
        if not (0 <= m < self.n):
            raise ValueError("message m 必須在 [0, N)")
        if r is None:
            r = rand_coprime(self.n)  # 近似 Z*_N

        c1 = ((1 + m * self.n) % self.n2) * pow(h, r, self.n2) % self.n2
        c2 = pow(self.g_core, r, self.n2)
        return (c1, c2)

    def strong_decrypt(self, c1: int) -> int:
        """強私鑰解密：論文公式 m = L(C1^λ mod N^2) * μ (mod N) """
        u = pow(c1, self.lambda_dec, self.n2)
        return (L(u, self.n) * self.mu) % self.n

    def weak_decrypt(self, c1: int, c2: int, theta_i: int) -> int:
        """弱私鑰解密：剝除 h^r，需要乘上 (C2^theta) 的模逆"""
        inv = modinv(pow(c2, theta_i, self.n2), self.n2)
        val = (c1 * inv) % self.n2
        return L(val,self.n)

    def ciphertext_refresh(self, c: Cipher, h: int, r_prime: int | None = None) -> Cipher:
        """ 重新隨機化（明文不變）"""
        if r_prime is None:
            r_prime = rand_coprime(self.n)
        c1, c2 = c
        c1p = (c1 * pow(h, r_prime, self.n2)) % self.n2
        c2p = (c2 * pow(self.g_core, r_prime, self.n2)) % self.n2
        return (c1p, c2p)

    @staticmethod
    def homomorphic_add(ca: Cipher, cb: Cipher, n2: int) -> Cipher:
        """兩個同一公鑰下的密文"""
        c1a, c2a = ca
        c1b, c2b = cb
        return ((c1a * c1b) % n2, (c2a * c2b) % n2)
    def homomorphic_scalar_multiply(self, cipher: Cipher, k: int) -> Cipher:
        c1, c2 = cipher
        return (pow(c1, k, self.n2), pow(c2, k, self.n2))


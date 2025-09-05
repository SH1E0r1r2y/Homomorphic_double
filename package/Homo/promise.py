# package/Homo/promise.py
import random
from typing import List, Tuple
from package.Homo.utils import modinv, next_prime, is_probable_prime

class PedersenVSS:
    """PedersenVSS"""
    def __init__(self, p: int, q: int, alpha: int, beta: int):
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def keygen(cls, min_q_bits: int = 256) -> "PedersenVSS":
        """
        產生 safe prime p = 2q + 1，並找 α、β（子群階 q 的生成元）。
        min_q_bits 建議 >= 祕密 bit 長度 + 一些 buffer。
        """
        q = next_prime(1 << (min_q_bits - 1) | 1)
        p = 2 * q + 1
        while not is_probable_prime(p):
            q = next_prime(q + 2)
            p = 2 * q + 1

        def pick_generator() -> int:
            # 取 g，再做 g^((p-1)/q) 投影進子群，避免得到 1
            while True:
                g = random.randrange(2, p - 1)
                cand = pow(g, (p - 1) // q, p)
                if cand != 1:
                    return cand
       
        alpha = pick_generator()
        beta = pick_generator()
        while beta == alpha:
            beta = pick_generator()

        return cls(p, q, alpha, beta)

    def commit(self, a: int, b: int) -> int:
        """E(a,b) = α^a β^b mod p"""
        return (pow(self.alpha, a, self.p) * pow(self.beta, b, self.p)) % self.p

    def init(self, s: int, t: int, d: int):
        """
        Initialization phase
        - 祕密 s ∈ Z_q
        - 門檻 t
        - d 個參與者
        """
        v = random.randrange(1, self.q)
        # f(x) = s + a1 x + ... + a_{t-1} x^{t-1}
        a_coeffs = [s] + [random.randrange(0, self.q) for _ in range(t - 1)]
        # g(x) = v + b1 x + ... + b_{t-1} x^{t-1}
        b_coeffs = [v] + [random.randrange(0, self.q) for _ in range(t - 1)]

        # 承諾
        e0 = self.commit(a_coeffs[0], b_coeffs[0])
        es = [self.commit(a_coeffs[j], b_coeffs[j]) for j in range(1, t)]

        shares = []
        for i in range(1, d + 1):
            si = sum(a_coeffs[j] * pow(i, j, self.q) for j in range(t)) % self.q
            vi = sum(b_coeffs[j] * pow(i, j, self.q) for j in range(t)) % self.q
            shares.append((i, si, vi))

        return e0, es, shares

    def verify(self, i: int, si: int, vi: int, e0: int, es: List[int], t: int) -> bool:
        """E(si,vi) ?= ∏_{j=0}^{t-1} E_j^{i^j}"""
        left = self.commit(si, vi)
        right = 1
        for j in range(t):
            ej = e0 if j == 0 else es[j - 1]
            right = (right * pow(ej, pow(i, j, self.q), self.p)) % self.p
        return left == right

    def recover(self, shares: List[Tuple[int, int]], t: int) -> int:
        """
        用前 t 份 share 做拉格朗日插值 (x=0)重建
        shares: [(i, si)]
        """
        xs = [i for (i, _) in shares]
        ys = [si for (_, si) in shares]

        def lagrange_basis_at_zero(k: int) -> int:
            xk = xs[k]
            num, den = 1, 1
            for j, xj in enumerate(xs):
                if j == k:
                    continue
                num = (num * xj) % self.q
                den = (den * (xj - xk)) % self.q
            return (num * modinv(den, self.q)) % self.q

        s = 0
        for k in range(len(xs)):
            s = (s + ys[k] * lagrange_basis_at_zero(k)) % self.q
        return s

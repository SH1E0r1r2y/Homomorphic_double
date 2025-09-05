# package/Homo/promise.py
import random
from typing import List, Tuple
from .utils import modinv

class PedersenVSS:
    def __init__(self, p: int, q: int, alpha: int, beta: int):
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta

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
        E0 = self.commit(a_coeffs[0], b_coeffs[0])
        Es = [self.commit(a_coeffs[j], b_coeffs[j]) for j in range(1, t)]

        # 分享
        shares = []
        for i in range(1, d + 1):
            si = sum(a_coeffs[j] * pow(i, j, self.q) for j in range(t)) % self.q
            vi = sum(b_coeffs[j] * pow(i, j, self.q) for j in range(t)) % self.q
            shares.append((i, si, vi))

        return E0, Es, shares

    def verify(self, i: int, si: int, vi: int, E0: int, Es: List[int], t: int) -> bool:
        """驗證 E(si,vi) ?= ∏_{j=0}^{t-1} E_j^{i^j}"""
        left = self.commit(si, vi)
        right = 1
        for j in range(t):
            Ej = E0 if j == 0 else Es[j - 1]
            right = (right * pow(Ej, pow(i, j, self.q), self.p)) % self.p
        return left == right

    def recover(self, shares: List[Tuple[int, int]], t: int) -> int:
        """
        秘密重建：用前 t 份 share 做拉格朗日插值 (x=0)
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

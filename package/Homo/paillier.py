# paillier.py
import random
from math import lcm
from typing import Dict, Tuple,List
from package.Homo.promise import PedersenVSS
from package.Homo.utils import modinv,L, rand_coprime, generate_strong_primes, find_g

Cipher = Tuple[int, int]  # (C1, C2)

class Paillier:
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
        p, q = generate_strong_primes(k)
        lambda_dec  = lcm(p - 1, q - 1)//2
        g_core, n, n2 = find_g(p, q, lambda_dec)
        # μ = (L(g^λ mod N^2))^{-1} mod N
        lu = L(pow((n + 1) % n2, lambda_dec, n2), n)
        mu = pow(lu, -1, n)
        theta_ta = random.randint(1, n // 4)
        h_ta = pow(g_core, theta_ta, n2)
        return cls(n=n, n2=n2, g_core=g_core, lambda_dec=lambda_dec, theta_ta=theta_ta, h_ta=h_ta, mu=mu)

    def encrypt(self, m: int, h: int, r: int | None = None) -> Cipher:
        if not (0 <= m < self.n):
            raise ValueError("message m 必須在 [0, N)")
        if r is None:
            r = rand_coprime(self.n)  # 近似 Z*_N

        c1 = ((1 + m * self.n) % self.n2) * pow(h, r, self.n2) % self.n2
        c2 = pow(self.g_core, r, self.n2)
        return (c1, c2)

    def strong_decrypt(self, c: Tuple[int,int], lambda_override: int = None) -> int:
        c1, c2 = c
        λ = lambda_override if lambda_override is not None else self.lambda_dec
        u = pow(c1, λ, self.n2)
        return (L(u, self.n) * self.mu) % self.n

    def weak_decrypt(self, c1: int, c2: int, theta_i: int) -> int:
        inv = modinv(pow(c2, theta_i, self.n2), self.n2)
        val = (c1 * inv) % self.n2
        return L(val,self.n)

    def refresh(self, c: Cipher, h: int, r_prime: int | None = None) -> Cipher:
        if r_prime is None:
            r_prime = rand_coprime(self.n)
        c1, c2 = c
        c1p = (c1 * pow(h, r_prime, self.n2)) % self.n2
        c2p = (c2 * pow(self.g_core, r_prime, self.n2)) % self.n2
        return (c1p, c2p)

    def homomorphic_scalar_multiply(self, cipher: Cipher, k: int) -> Cipher:
        c1, c2 = cipher
        return (pow(c1, k, self.n2), pow(c2, k, self.n2))
    
    @staticmethod
    def homomorphic_multiply(ca: tuple, cb: tuple, n2: int) -> tuple:
        """
        密文 × 密文 (同態加法性質)：E(m1) * E(m2) = E(m1 + m2)
        """
        c1a, c2a = ca
        c1b, c2b = cb
        return ((c1a * c1b) % n2, (c2a * c2b) % n2)

    @staticmethod
    def homomorphic_subtract(ca: tuple, cb: tuple, n2: int) -> tuple:
        """
        同態減法：E(m1) * E(-m2) = E(m1 - m2)
        """
        c1a, c2a = ca
        c1b, c2b = cb
        # pow(c2b, n2-1, n2) 相當於取負數 (模 n2)
        return ((c1a * pow(c1b, n2-1, n2)) % n2,
                (c2a * pow(c2b, n2-1, n2)) % n2)

    @staticmethod
    def homomorphic_add(ca: tuple, cb: tuple, n2: int) -> tuple:
        """
        同態加法：E(m1) * E(m2) = E(m1 + m2)
        """
        c1a, c2a = ca
        c1b, c2b = cb
        return ((c1a * c1b) % n2, (c2a * c2b) % n2)

class FunctionNode:
    def __init__(self, id: int, paillier: Paillier,
                 lambda_share: int, theta_share: int,
                 pedersen_commit: int,
                 v_i: int, v_i_prime: int):
        self.id = id
        self.paillier = paillier
        self.lambda_share = lambda_share  # λ_i
        self.theta_share = theta_share    # θ_i
        self.pedersen_commit = pedersen_commit  # α^s_i * β^v_i
        self.v_i = v_i            # λ 的 Pedersen 隨機數
        self.v_i_prime = v_i_prime  # θ 的 Pedersen 隨機數


    def verify_share(self, vss: PedersenVSS, e0_l: int, es_l: List[int], e0_t: int, es_t: List[int], t: int) -> bool:
        """
        使用 PedersenVSS 驗證 share 是否正確
        - e0_l, es_l: λ 的 Pedersen 承諾
        - e0_t, es_t: θ 的 Pedersen 承諾
        """
        # 驗證 λ_share
        valid_lambda = vss.verify(self.id, self.lambda_share, self.v_i, e0_l, es_l, t)
        # 驗證 θ_share
        valid_theta  = vss.verify(self.id, self.theta_share, self.v_i_prime, e0_t, es_t, t)
        
        return valid_lambda and valid_theta

    def refresh(self, cipher_pair: Tuple[int,int]) -> Tuple[int,int]:
        """
        1) 使用節點弱私鑰 θ_i 部分解密
        2) 再用系統公鑰隨機加密生成新密文
        """
        c1, c2 = cipher_pair

        # 部分解密
        partial_plain = self.paillier.weak_decrypt(c1, c2, self.theta_share)

        # 隨機再加密
        r_new = random.randint(1, self.paillier.n // 4)
        c1_new, c2_new = self.paillier.encrypt(partial_plain, self.paillier.h_ta, r_new)

        return (c1_new, c2_new)

    def partial_decrypt(self, cipher_pair: Tuple[int, int]) -> Tuple[int, int]:
        """使用 λ_i 對 ciphertext 做部分解密"""
        c1, c2 = cipher_pair
        return (
            pow(c1, self.lambda_share, self.paillier.n2),
            pow(c2, self.lambda_share, self.paillier.n2),
        )
        
    # @staticmethod
    # def combine_shares(
    #     partials: List[Tuple[int, List[Tuple[int, int]]]],
    #     n2: int
    # ) -> List[Tuple[int, int]]:
    #     """
    #     合併多個節點的部分解密結果，重建完整明文 (僅在排序階段需要)
    #     partials: [(node_id, [(c1^λᵢ, c2^λᵢ), ...]), ...]
    #     返回 [(c1, c2), ...] → 可還原 TF 值
    #     """
    #     ids = [pid for pid, _ in partials]

    #     def lagrange_coeff(j):
    #         num, den = 1, 1
    #         for m in ids:
    #             if m != j:
    #                 num = (num * (-m)) % n2
    #                 den = (den * (j - m)) % n2
    #         return num * pow(den, -1, n2) % n2

    #     part_dict = {pid: vec for pid, vec in partials}
    #     length = len(next(iter(part_dict.values())))
    #     result = []
    #     for idx in range(length):
    #         acc1, acc2 = 1, 1
    #         for j, vec in part_dict.items():
    #             lj = lagrange_coeff(j)
    #             c1j, c2j = vec[idx]
    #             acc1 = (acc1 * pow(c1j, lj, n2)) % n2
    #             acc2 = (acc2 * pow(c2j, lj, n2)) % n2
    #         result.append((acc1, acc2))
    #     return result


class Entity:
    def __init__(self, id: int, pk: Tuple[int, int, int], sk_weak: int):
        self.id = id
        self.pk = pk  # (N, g_core, h)
        self.sk_weak = sk_weak  # theta_i

    @staticmethod
    def gen_entity_key(n: int, g_core: int, n2: int, theta: int | None = None
                      ) -> Dict[str, Tuple[int, int, int]]:
        if theta is None:
            theta = random.randint(1, n // 4)
        
        h = pow(g_core, theta, n2)
        
        return {"pk": (n, g_core, h), "sk_weak": theta}
    
    @classmethod
    def register_data_owner(cls, paillier, entity_id: str = None):
        if entity_id is None:
            entity_id = f"DO_{random.randint(1, 999):03d}"
        
        print(f"[*] TA 正在註冊 Data Owner: {entity_id}")
        
        do_keys = cls.gen_entity_key(paillier.n, paillier.g_core, paillier.n2)
        do = cls(id=entity_id, pk=do_keys["pk"], sk_weak=do_keys["sk_weak"])
        
        print(f"     ✓ DO 註冊成功，金鑰已建立")
        
        return do

    @classmethod
    def register_data_user(cls, paillier, entity_id: str = None):
        if entity_id is None:
            entity_id = f"DU_{random.randint(1, 999):03d}"
            
        print(f"[*] TA 正在註冊 Data User: {entity_id}")
        
        du_keys = cls.gen_entity_key(paillier.n, paillier.g_core, paillier.n2)
        du = cls(id=entity_id, pk=du_keys["pk"], sk_weak=du_keys["sk_weak"])
        
        print(f"  ✓ DU 註冊成功，金鑰已建立")
        
        return du

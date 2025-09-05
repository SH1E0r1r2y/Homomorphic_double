import random
from math import gcd
from secrets import randbelow
from sympy import randprime, isprime

def L(u: int, n: int) -> int:
    return (u - 1) // n
    
def modinv(a: int, mod: int) -> int:
    return pow(a, -1, mod)

def rand_coprime(modulus: int) -> int:
    #取與 modulus 互質的整數，用來找隨機數 r ∈ Z*_{N}
    if modulus <= 3:
        raise ValueError("modulus 太小")
    while True:
        # 產生 2..modulus-2 的均勻亂數
        x = 2 + randbelow(modulus - 3)
        if gcd(x, modulus) == 1:
            return x

def generate_strong_primes(bit_length: int):
    while True:
        p = randprime(2**(bit_length - 1), 2**bit_length)
        q = randprime(2**(bit_length - 1), 2**bit_length)
        if p != q and isprime((p - 1) // 2) and isprime((q - 1) // 2):
            return p, q

def find_g(p: int, q: int, lambda_dec: int):
    N  = p * q
    N2 = N * N
    p_ = (p - 1) // 2
    q_ = (q - 1) // 2
    lam_g = lambda_dec // 2  # g 的階 = lcm(p', q')
    target = 2 * p_ * q_

    while True: # 直到挑到適合的 g
        while True: # 先找到與 N^2 互質的 a，隨機 a ∈ Z*_{N^2}，令 g ≡ -a^{2N} (mod N^2)
            a = random.randrange(2, N2 - 1)
            if gcd(a, N2) == 1:
                break
        g = (-pow(a, 2 * N, N2)) % N2 #-a^2N mod N^2
        if g == 1:
            continue  # trivial，只能生成自己{1}，故重抽

        # A) gcd(L(g^λ mod N^2), N) == 1，確認g的可逆μ存在 
        u = pow(g, lam_g, N2)
        Lu = L(u, N)
        if gcd(Lu, N) != 1:
            continue

        # B) 確認g可生成大子群=2*p'*q'，且沒有被困於小子群p',q',2
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

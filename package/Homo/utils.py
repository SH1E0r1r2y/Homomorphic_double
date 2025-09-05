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
    n2 = N * N
    p_ = (p - 1) // 2
    q_ = (q - 1) // 2
    lam_g = lambda_dec // 2  # g 的階 = lcm(p', q')
    target = 2 * p_ * q_

    while True: # 直到挑到適合的 g
        while True: # 先找到與 N^2 互質的 a，隨機 a ∈ Z*_{N^2}，令 g ≡ -a^{2N} (mod N^2)
            a = random.randrange(2, n2 - 1)
            if gcd(a, n2) == 1:
                break
        g = (-pow(a, 2 * N, n2)) % n2 #-a^2N mod N^2
        if g == 1:
            continue  # trivial，只能生成自己{1}，故重抽

        # A) gcd(L(g^λ mod N^2), N) == 1，確認g的可逆μ存在 
        u = pow(g, lam_g, n2)
        lu = L(u, N)
        if gcd(lu, N) != 1:
            continue

        # B) 確認g可生成大子群=2*p'*q'，且沒有被困於小子群p',q',2
        if pow(g, target, n2) != 1:
            continue

        bad = False
        for r in (2, p_, q_):
            if pow(g, target // r, n2) == 1:
                bad = True
                break
        if bad:
            continue

        return g, N, n2


def is_probable_prime(n: int) -> bool:
    if n < 3 or n % 2 == 0:
        return n == 2
    # 固定 bases：對 64-bit 以上也很實用；若要超大整數可再加 bases
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if a >= n:
            continue
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def next_prime(n: int) -> int:
    if n <= 2:
        return 2
    n = n + 1 if n % 2 == 0 else n
    while not is_probable_prime(n):
        n += 2
    return n
import random
from sympy import randprime, isprime, lcm

def initialize_TA():
    bit_length = 32
    while True:
        q = randprime(2**(bit_length - 2), 2**(bit_length - 1))  # 約 32-bit
        p = randprime(2**(bit_length - 2), 2**(bit_length - 1))
        if isprime((p-1)//2) and isprime((q-1)//2):
            break
        n = p * q
        n2 = n * n
        lambda_sys = lcm(p-1,q-1)//2
        theta_sys = random.randint(1, n // 4)

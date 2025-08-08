class Paillier:
    def __init__(self, bit_length):
        self.bit_length = bit_length

    def L(self, u):
        return (u - 1) / self.n
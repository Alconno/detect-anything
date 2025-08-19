import math

def pairing_function(a: int, b: int) -> int:
    a, b = int(a), int(b)
    return ((a + b + 1) * (a + b)) // 2 + b

def reverse_pairing_function(z: int) -> tuple[int, int]:
    z = int(z) 
    w = (math.isqrt(8 * z + 1) - 1) // 2
    t = (w * (w + 1)) // 2
    y = z - t
    x = w - y
    return x, y

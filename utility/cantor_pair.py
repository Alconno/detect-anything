import math

def pairing_function(a: int, b: int) -> int:
    """Encode two integers into a single unique integer."""
    return ((a + b + 1) * (a + b)) // 2 + b

def reverse_pairing_function(z: int) -> tuple[int, int]:
    """Decode the single integer back into the original two integers."""
    w = (math.isqrt(8 * z + 1) - 1) // 2
    t = (w * (w + 1)) // 2
    y = z - t
    x = w - y
    return x, y

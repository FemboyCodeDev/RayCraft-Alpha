import numpy as np


def fade(t):
    """Smoothing function to make the terrain look natural."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, x):
    """Linear interpolation."""
    return a + x * (b - a)


def noise_4d(x, y, z, w, seed=42):
    """Generates a single noise value for 4D coordinates."""
    # Create a repeatable random state based on the seed
    rng = np.random.default_rng(seed)

    # 1. Identify the 'cell' in 4D space
    X, Y, Z, W = int(x), int(y), int(z), int(w)
    xf, yf, zf, wf = x - X, y - Y, z - Z, w - W

    # 2. Fade curves for each dimension
    u, v, p, q = fade(xf), fade(yf), fade(zf), fade(wf)

    # 3. Create a 16-corner hypercube of random values
    # In a real Perlin implementation, we'd use gradients,
    # but for a 'from scratch' base, we'll interpolate 16 random corners.
    points = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # Deterministic hash for the corner
                    corner_seed = hash((X + i, Y + j, Z + k, W + l, seed)) % (2 ** 32)
                    points[i, j, k, l] = np.random.default_rng(corner_seed).random()

    # 4. Interpolate along the 4 dimensions (The 'Nested' Lerp)
    # Interpolate X
    x000 = lerp(points[0, 0, 0, 0], points[1, 0, 0, 0], u)
    x001 = lerp(points[0, 0, 0, 1], points[1, 0, 0, 1], u)
    x010 = lerp(points[0, 0, 1, 0], points[1, 0, 1, 0], u)
    x011 = lerp(points[0, 0, 1, 1], points[1, 0, 1, 1], u)
    x100 = lerp(points[0, 1, 0, 0], points[1, 1, 0, 0], u)
    x101 = lerp(points[0, 1, 0, 1], points[1, 1, 0, 1], u)
    x110 = lerp(points[0, 1, 1, 0], points[1, 1, 1, 0], u)
    x111 = lerp(points[0, 1, 1, 1], points[1, 1, 1, 1], u)

    # Interpolate Y
    y00 = lerp(x000, x100, v)
    y01 = lerp(x001, x101, v)
    y10 = lerp(x010, x110, v)
    y11 = lerp(x011, x111, v)

    # Interpolate Z
    z0 = lerp(y00, y10, p)
    z1 = lerp(y01, y11, p)

    # Interpolate W (The 4th dimension!)
    return lerp(z0, z1, q)


# Example usage:
val = noise_4d(1.5, 2.2, 0.9, 5.5)
print(f"The noise value at this 4D point is: {val}")
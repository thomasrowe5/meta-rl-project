import numpy as np

def generate_test_scenarios(n):
    """Return n dummy state arrays"""
    return [np.random.rand(4) for _ in range(n)]

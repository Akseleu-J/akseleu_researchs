"""Segment folding: n independent segments become the batch axis (one kernel call, no Python loop)."""


def fold(t, n, C):
    """(B,H,n*C,D) -> (B*n,H,C,D)"""
    B, H, L, D = t.shape
    return t.reshape(B, H, n, C, D).transpose(0, 2, 1, 3, 4).reshape(B * n, H, C, D)


def unfold_out(o, B, n):
    """(B*n,H,C,D) -> (B,H,n*C,D)"""
    BN, H, C, D = o.shape
    return o.reshape(B, n, H, C, D).transpose(0, 2, 1, 3, 4).reshape(B, H, n * C, D)


def unfold_states(S, B, n):
    """(B*n,H,D,E) -> (B,H,n,D,E)"""
    BN, H, D, E = S.shape
    return S.reshape(B, n, H, D, E).transpose(0, 2, 1, 3, 4)

import numpy as np
from .D_lambda_K import D_lambda_K
# from .D_s import D_s
from .my_D_s import D_s

def HQNR(ps_ms, ms, msexp, pan, S, sensor, ratio):
    """HQNR index and maps

    Args:
        ps_ms (ndarray): fusion
        ms (ndarray): MS_LR
        msexp (ndarray): MS
        pan (ndarray): pan
        S (int): block size
        sensor (str): sensor type
        ratio (int): fusion ratio

    Returns:
        [float, float, float, ndarray, ndarray, ndarray]: indices and maps
    """
    D_lambda, D_l_map = D_lambda_K(ps_ms, msexp, ratio, sensor, S) # .copy()
    D_S, D_S_map = D_s(ps_ms, msexp, ms, pan, ratio, S, 1)
    HQNR_index = (1 - D_lambda) * (1 - D_S)
    HQNR_map = (1 - D_l_map) * (1 - D_S_map.mean(-1))

    return HQNR_index, D_lambda, D_S, HQNR_map

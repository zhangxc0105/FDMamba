from .MTF import MTF
from .my_q2n import q2n

def D_lambda_K(fused,ms,ratio,sensor,S):

    if (fused.shape[0] != (ms.shape[0]) or fused.shape[1] != ms.shape[1]) == 1:
        print("The two images must have the same dimensions")
        return -1
    
    # N = fused.shape[0]
    # M = fused.shape[1]
    # if np.remainder(N,S-1) != 0:
    #     print("Number of rows must be multiple of the block size")
    #     return -1    
    # if np.remainder(M,S-1) != 0:
    #     print("Number of columns must be multiple of the block size")
    #     return -1

    fused_degraded = MTF(fused,sensor,ratio)
                
    # fused_degraded = fused_degraded[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio),:]
        
    Q2n_index, q2n_map = q2n(ms, fused_degraded, S, S)
        
    Dl = 1 - Q2n_index
    Dl_map = 1 - q2n_map
    
    return Dl, Dl_map

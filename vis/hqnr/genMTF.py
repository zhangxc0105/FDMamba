import numpy as np
from .tools import fir_filter_wind, gaussian2d, kaiser2d

def genMTF(ratio, sensor, nbands):
    
    N = 41
        
    if sensor=='QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1' or sensor=='WV4':
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='WV2':
        GNyq = [0.35*np.ones(nbands),0.27]
    elif sensor=='WV3':
        GNyq = np.asarray([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315],dtype='float32') 
    else:
        GNyq = 0.3 * np.ones(nbands)
        
    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1/ratio

    h = np.zeros((N,N,nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[ii])))
        H=gaussian2d(N,alpha)
        Hd=H/np.max(H)
        w=kaiser2d(N,0.5)
        h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
        
    return h
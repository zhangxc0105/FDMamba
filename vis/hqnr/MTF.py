from scipy import ndimage
import numpy as np
from hqnr.genMTF import genMTF

def MTF(I_MS,sensor,ratio):
    
    h = genMTF(ratio, sensor,I_MS.shape[2])
    
    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:,:,ii] = ndimage.filters.correlate(I_MS[:,:,ii],h[:,:,ii],mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)
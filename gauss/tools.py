import numpy as np

class FourierSmooth(object):
    def __init__(self, n=5):
        self.n = n

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        rft[self.n:] = 0
        return np.fft.irfft(rft)

def count_maxs(feature_i):
    extr_i=local_extr(feature_i)
    n_max= np.where(extr_i==(-2))[0].shape[0]
    return n_max

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))
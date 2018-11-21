import numpy as np
#numpy.random
from sklearn.mixture import GaussianMixture
import basic.extr

def gausian_feats(ts):
#    feat_i=np.expand_dims(feat_i,axis=0).T
    print(ts.shape)
    mixture=GaussianMixture(n_components=3,
                   covariance_type="full", max_iter=20, random_state=0)
    samples=rejection_sampling(ts,size=500)
    print(samples.shape)
    samples=np.expand_dims(samples,axis=0).T
    mixture.fit(samples)
    means=np.squeeze(mixture.means_) 
    covars=np.squeeze(mixture.covariances_)
    return list(means) + list(covars)

def rejection_sampling(feats,size=1000):
    seq_len=feats.shape[0]
    feats-=np.min(feats)
    feats/=np.max(feats)
    samples=np.arange(seq_len)/float(seq_len-1)
    indexes=np.random.randint(seq_len,size=size)
    threshold=np.random.uniform( size=size)
    return np.array([ samples[i] for i in indexes
                        if( feats[i]>threshold[i])])

extract=basic.extr.Extractor(gausian_feats)

extract( "../LSTM/norm_full", "test")

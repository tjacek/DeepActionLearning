import basic
import numpy as np
from sklearn.cluster import KMeans

def cluster_quality(dataset):
    if(type(dataset)==str or type(dataset)==list):
        dataset=basic.read_dataset(dataset)
    train_data=dataset.split()[0]
    labels=np.array(train_data.y)
    cats=np.unique(labels)
    cls_labels=cluster(train_data)
    cls_labels+=1
    def cls_helper(cat_i):
    	indic_i=indicator_vector(cat_i,labels)
        cls_indic=cls_labels*indic_i
        counted_i=np.bincount(cls_indic)
        counted_i[0]=0
        in_cls=np.max(counted_i)
        cat_size=np.sum(indic_i)
        return float(in_cls)/float(cat_size)
    return [ cls_helper(cat_i) for cat_i in cats]

def cluster(dataset):
    n_clust=dataset.n_cats()
    kmeans= KMeans(n_clusters=n_clust,random_state=0)
    kmeans.fit(dataset.X)
    return kmeans.labels_

def indicator_vector(i,y):
    ind_vector=np.zeros(y.shape,dtype=int)
    ind_vector[y==i]=1.0
    return ind_vector
import basic
import numpy as np
from sklearn.cluster import KMeans

def cluster_quality(dataset):
    if(type(dataset)==str or type(dataset)==list):
        dataset=basic.read_dataset(dataset)
    train_data=dataset.split()[0]
    labels=np.array(train_data.y)
    cats=np.unique(labels)
    cls_labels,cls_indicators=get_clusters(train_data)
    def cls_helper(cat_i):
    	indic_i=indicator_vector(cat_i,labels)
        cat_size=np.sum(indic_i)
        cls_labels_i=cls_labels*indic_i
        cls_i,in_cls=asign_cluster(cls_labels_i)
        cls_size=np.sum(cls_indicators[cls_i])
        prec=float(in_cls)/float(cls_size)
        recall=float(in_cls)/float(cat_size)
        return 2*(prec*recall)/(prec+recall)#float(in_cls)/float(cls_size)
    return [ cls_helper(cat_i) for cat_i in cats]

def asign_cluster(cls_i):
    counted_i=np.bincount(cls_i)
    counted_i[0]=0
    return np.argmax(counted_i)-1,np.max(counted_i)

def get_clusters(train_data):
    cls_labels=clustering(train_data)
    n_cls=np.amax(cls_labels)+1
    cls_indicators=[indicator_vector(i,cls_labels) 
                        for i in range(n_cls)]
    cls_labels+=1
    return cls_labels,cls_indicators

def clustering(dataset):
    n_clust=dataset.n_cats()
    kmeans= KMeans(n_clusters=n_clust,random_state=0)
    kmeans.fit(dataset.X)
    return kmeans.labels_

def indicator_vector(i,y):
    ind_vector=np.zeros(y.shape,dtype=int)
    ind_vector[y==i]=1.0
    return ind_vector
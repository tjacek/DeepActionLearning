import sys,os
sys.path.append(os.path.abspath('../cluster_images'))
import numpy as np 
from collections import Counter
import seq
import utils.paths as paths
from utils.timer import clock 
import utils.data

@paths.path_args
def use_dtw(dataset_path,k=0,dataset_format='cp_dataset',select_type='modulo'):
    test,train=seq.seq_dataset(dataset_path,dataset_format=dataset_format)
    wrap=Wrap()
    print(len(train['y']))
    y_pred=wrap(train,test)
    seq.check_prediction(y_pred,test['y'])
    wrap.show()
    
class Wrap(object):
    def __init__(self):
        self.results={}
    
    @clock
    def __call__(self,train,test): 
        def knn_helper(i,test_i):
            name_i=test['names'][i]
            cat_true_i=test['y'][i]
            cat_i=knn(test_i,train) 
            self.results[name_i]=(cat_i,cat_true_i)
            return cat_i
        return [knn_helper(i,test_i) 
                 for i,test_i in enumerate(test['x'])]

    def show(self):
        for key_i,value_i in self.results.items():
            print(key_i + " %d"  % value_i)

def knn(new_x,train_dataset,k=1):
    distance=[dtw_metric(new_x,x_i) 
              for x_i in train_dataset['x']]
    dist_inds=get_dist_inds(distance,k)
    y=   train_dataset['y']
    
    nearest=[y[i] for i in dist_inds]
    print(train_dataset['x'][0].shape)
    print(nearest)
    print(dist_inds)
    new_cat=most_common(nearest)
    print(new_cat)
    return new_cat

def get_k_distances(distances,dist_inds):
    return [distances[i]  for i in dist_inds]

def get_dist_inds(distance,k=1):
    distance=np.array(distance)
    return distance.argsort()[0:k]

def most_common(nearest):
    count =Counter(nearest)
    return count.most_common()[0][0]    

def dtw_metric(s,t):
    n=len(s)
    m=len(t)
    dwt=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        dwt[i][0]=np.inf
    for i in range(1,m+1):
        dwt[0][i]=np.inf
    dwt[0][0]=0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=d1(s[i-1],t[j-1])
            dwt[i,j]=cost+min([dwt[i-1][j],dwt[i][j-1],dwt[i-1][j-1]])
    return dwt[n][m]

def d1(v,d):
    return np.linalg.norm(v-d)

def d2(v,u):
    dist=np.dot(u,v)
    dist/=np.linalg.norm(v) * np.linalg.norm(u)
    return dist

if __name__ == "__main__":
    path= "../../Documents/FF/seq" #'../../AA_dtw2/max_z/seq'
    use_dtw(path,0,'basic_dataset')
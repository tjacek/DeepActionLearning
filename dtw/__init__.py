import numpy as np

def dtw_basic(s,t):
    dtw,n,m=prepare_matrix(s,t)
    for i in range(1,n+1):
        for j in range(1,m+1):
            t_i,t_j=s[i-1],t[j-1]
            diff=t_i-t_j
            cost= np.dot(diff,diff)#np.sum((t_i-t_j)**2)           
            dtw[i][j]=cost+min([dtw[i-1][j],dtw[i][j-1],dtw[i-1][j-1]])
    return np.sqrt(dtw[n][m])

def dtw_optim(s,t,w=50):
    dtw,n,m=prepare_matrix(s,t)
    w=max(w,np.abs(n-m))
    for i in range(1,n+1):
        start_i,end_i=max(1,i-w),min(m+1,i+w)
        for j in range(start_i,end_i):
            t_i,t_j=s[i-1],t[j-1]
            diff=t_i-t_j
            cost= np.dot(diff,diff) #np.sum((t_i-t_j)**2)   
            dtw[i][j]=cost+min([dtw[i-1][j],dtw[i][j-1],dtw[i-1][j-1]])
    return np.sqrt(dtw[n][m])

def prepare_matrix(s,t):
    n=len(s)
    m=len(t)
    cost_matrix=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        cost_matrix[i][0]=np.inf
    for i in range(1,m+1):
        cost_matrix[0][i]=np.inf
    return cost_matrix,n,m    

def d1(v,d):
    return np.linalg.norm(v-d)

def d2(v,u):
    dist=np.dot(u,v)
    dist/=np.linalg.norm(v) * np.linalg.norm(u)
    return dist

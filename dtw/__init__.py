import numpy as np

def dtw_basic(s,t):
    dtw,n,m=prepare_matrix(s,t)
    for i in range(1,n+1):
        for j in range(1,m+1):
            t_i,t_j=s[i-1],t[j-1]
            cost=(t_i-t_j)**2           
            dwt[i,j]=cost+min([dwt[i-1][j],dwt[i][j-1],dwt[i-1][j-1]])
    return np.sqrt(dwt[n][m])

def dtw_optim(s,t,w=10):
    dtw,n,m=prepare_matrix(s,t)
    w=max(w,np.abs(n-m))
    for i in range(1,n+1):
        start_i,end_i=max(1,i-w),min(m+1,i+w)
        for j in range(start_i,end_i):
            t_i,t_j=s[i-1],t[j-1]
            cost=(t_i-t_j)**2   
            dwt[i,j]=cost+min([dwt[i-1][j],dwt[i][j-1],dwt[i-1][j-1]])
    return np.sqrt(dwt[n][m])

def prepare_matrix(s,t):
    n=len(s)
    m=len(t)
    dwt=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        dwt[i][0]=np.inf
    for i in range(1,m+1):
        dwt[0][i]=np.inf
    return dtw,n,m    

def d1(v,d):
    return np.linalg.norm(v-d)

def d2(v,u):
    dist=np.dot(u,v)
    dist/=np.linalg.norm(v) * np.linalg.norm(u)
    return dist

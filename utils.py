import os,os.path,re,pickle
import itertools
import numpy as np 
import sklearn

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def bottom_files(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            paths=[ root+'/'+filename_i 
                for filename_i in filenames]
            all_paths+=paths
    all_paths.sort(key=natural_keys)        
    return all_paths

def bottom_dirs(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            all_paths.append(root)
    all_paths.sort(key=natural_keys) 
    return all_paths

def top_dirs(path):
    all_paths=[os.path.join(path,name_i) for name_i in os.listdir(path)]
    return [name_i for name_i in all_paths
                if(os.path.isdir(name_i))]

def split(actions,selector):
    train,test=[],[]
    for action_i in actions:
        if(selector(action_i)):
            train.append(action_i)
        else:
            test.append(action_i)
    return train,test

def switch_paths(new_path,paths):
    postfixes=[ path_i.split("/")[-1] for path_i in paths]
    return [new_path+"/"+postfix_i for postfix_i in postfixes]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_name(path_i):
    return path_i.split('/')[-1]

def extract_numbers(text):
    str_numb=re.findall(r'\d+',text)
    return [int(n) for n in str_numb]   

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'rb')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def save_string(path,string):
    if(type(string)==list):
        string="\n".join(string)
    file_str = open(str(path),'w')
    file_str.write(string)
    file_str.close()

def read_lines(in_path,sep=','):
    with open(in_path) as f:
        lines = f.readlines()
        return [ line_i.split(sep) 
                    for line_i in lines]

def str_to_vector(str,sep=","):
    return [float(cord_i)
                for cord_i in str.split(sep) ]

def all_equal(items,value):
    return all(x == items[0] for x in items)

def one_hot(cat_i,n):
    vote_i=np.zeros((n,))
    vote_i[cat_i]=1
    return vote_i 

def save_matrix(name,matrix):
    name=name.split(".")[0]
    name+=".csv"
    np.savetxt(name,matrix,fmt="%.4f",delimiter=",")

def set_prefix(name,prefix):
    name=name.split(".")[0]
    name+=prefix
    return name

def linear_reg(clf_acc,feats_quality,pred=False):
    clf_acc,feats_quality=np.array(clf_acc), np.array(feats_quality)
    regr=sklearn.linear_model.LinearRegression()
    feats_quality=feats_quality[:,np.newaxis]
    regr.fit(feats_quality,clf_acc)
    if(pred):
        return regr,regr.predict(feats_quality)
    else:    
        return regr

def all_subsets(seq):
    subsets=[[]]
    for i in range(1,len(seq)):
        subsets+=itertools.combinations(seq,i)
    subsets=[ list(ss) for ss in subsets ]
    subsets.append(seq)
    return subsets
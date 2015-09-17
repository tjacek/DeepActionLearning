import os,sys,re

def save_combine(paths,out_path):
    file_ = open(out_path, 'w')
    file_.write(combine(paths))
    file_.close()

def combine(paths):
    partial_data=map(read_dataset,paths)
    new_instances=[]
    for i,instance in enumerate(partial_data[0]):
        full_instance=""
        for j,partial in enumerate(partial_data):
	    full_instance+=partial_data[j][i][0]+","
        if(len(instance)>1):
            full_instance+="#"+instance[1]+"\n"
            new_instances.append(full_instance)
    return  reduce(lambda x,y: x+y, new_instances,"") 

def read_dataset(path):
    raw_file = open(path)
    raw=raw_file.read()
    raw=raw.split("\n")
    raw=map(lambda(x): x.split(",#"),raw)
    return raw 

save_combine(['arff/xy','arff/zx','arff/zy'],"full")

import os
from os import listdir
from os.path import isfile, join

def transform_data(in_path,out_path,transform_file): 
    all_files = get_all_files(in_path)
    all_files.sort()
    for filename in all_files:
        in_file=in_path+filename
        out_file=out_path+filename
        transform_file(in_file,out_file)

def raw_to_tensor(in_file,out_file):
    out_file=out_file.replace(".bin",".tensor")
    print(out_file)
    cmd="th read-action.lua " + in_file+" "+out_file
    os.system(cmd)

def to_nonzero(in_file,out_file):
    out_file=out_file.replace(".tensor",".nonzero")
    print(out_file)
    cmd="th nonzero-action.lua " + in_file+" "+out_file
    os.system(cmd)

def to_diff(in_file,out_file):
    out_file=out_file.replace(".nonzero",".diff")
    print(out_file)
    cmd="th action.lua " + in_file+" "+out_file
    os.system(cmd)

def get_all_files(path):
    return [ f for f in listdir(path) if isfile(join(path,f)) ]

def show_data(path,show,cond=None):
    all_files = get_all_files(path)
    if(cond!=None):
        selected_files=filter(cond,all_files)
    else:
        selected_files=all_files
    for filename in selected_files:
        full_filename=path+filename
        show(full_filename)

def get_category_filter(integer):
    return lambda action: extract_category(action)==integer

def extract_category(filename):
    prefix=filename.split("_")[0]
    if(prefix[0]!='a'):
        return -1
    category=prefix.replace("a","")    
    return int(category)

def show_action(action_file):
    print(action_file)
    cmd="qlua show-action.lua " + action_file
    os.system(cmd)

def show_dim(action_file):
    action_name=action_file.split("/")[-1]
    #print(extract_category(action_name))
    cmd="qlua show-dim.lua " + action_file
    os.system(cmd)

in_path="/home/user/Desktop/nonzero_data/"
out_path="/home/user/Desktop/diff_data/"

#raw_to_tensor(in_path+scene,out_path+scene)
#transform_data(in_path,out_path,to_diff)
show_data(out_path,show_dim,get_category_filter(20))

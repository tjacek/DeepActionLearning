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

def to_action_desc(in_file,out_file):
    out_file=out_file.replace(".diff",".desc")
    print(out_file)
    cmd="th action-desc.lua " + in_file+" "+out_file
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
        return None
    category=prefix.replace("a","")    
    return int(category)

def extract_person(filename):
    prefix=filename.split("_")[1]
    if(prefix[0]!='s'):
        return None
    person=prefix.replace("s","")    
    return int(person)

def show_action(action_file):
    print(action_file)
    cmd="qlua show-action.lua " + action_file
    os.system(cmd)

def show_dim(action_file):
    action_name=action_file.split("/")[-1]
    #print(extract_category(action_name))
    cmd="qlua show-dim.lua " + action_file
    os.system(cmd)

def split_dataset(source,train,test):
    all_files = get_all_files(source)
    for filename in all_files:
        old_path=source+filename
        if((extract_person(filename)%2)==1):
            new_path=train+filename
        else:
            new_path=test+filename
        print(old_path)
        print(new_path)
        os.system("cp "+old_path+" "+new_path)

path="/home/user/Desktop/"
in_path=path+"diff_data/"
out_path=path+"desc_data/"

#transform_data(in_path,out_path,to_action_desc)
#show_data(out_path,show_action,get_category_filter(19))

train=path+"dataset_1/train/"
test=path+"dataset_1/test/"
split_dataset(out_path,train,test)

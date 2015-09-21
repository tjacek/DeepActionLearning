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

def to_var(in_file,out_file):
    out_file=out_file.replace(".nonzero",".var")
    print(out_file)
    cmd="th action-variance.lua " + in_file+" "+out_file
    os.system(cmd)

def to_projection(in_file,out_file):
    out_file=out_file.replace(".nonzero","_zy.nonzero")
    print(out_file)
    cmd="th projection.lua " + in_file+" "+out_file
    os.system(cmd)

def to_scaled_data(in_file,out_file):
    out_file=out_file.replace(".nonzero",".space")
    print(out_file)
    cmd="th scale-space.lua " + in_file+" "+out_file
    os.system(cmd)

def to_vol_data(in_file,out_file):
    out_file=out_file.replace(".space",".time")
    print(out_file)
    cmd="th scale-time.lua " + in_file+" "+out_file
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

def create_dataset(dir_name,data_type="spatial"):
    filename=dir_name.split("/")[-2]
    out_data=dir_name.replace(filename+"/",filename+".tensor")

    out_labels=dir_name.replace(filename+"/",filename+"_labels.tensor")
    #out_labels=out_labels.replace("/","")
    print(out_labels)
    cmd="th create-dataset.lua "
    cmd+=dir_name+" "+out_data+" "+out_labels+" "+data_type
    os.system(cmd)

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path) 

def generate_complet_dataset(in_path,out_path,data_type):
    make_dir(out_path)
    train=out_path+"train/"
    test=out_path+"test/"
    make_dir(train)
    make_dir(test) 
    split_dataset(in_path,train,test)
    create_dataset(train,data_type)
    create_dataset(test,data_type)

def generate_spatial_data(in_path,out_path):
    proj_path=out_path+"proj/"
    make_dir(proj_path)
    transform_data(in_path,proj_path, to_projection)
    var_path=out_path+"var/"
    make_dir(var_path)
    transform_data(proj_path,var_path,to_var)
    generate_complet_dataset(var_path,out_path,"spatial")

def generate_volumetric_data(in_path,out_path):
    generate_complet_dataset(in_path,out_path,"volumetric")

path="/home/user/Desktop/"
in_path=path+"time/"
out_path=path+"dataset_4/"
generate_volumetric_data(in_path,out_path)
#generate_complet_dataset(in_path,path+"dataset_4/")
#generate_spatial_data(in_path,out_path)

#transform_data(in_path,out_path, to_vol_data)
#show_data(in_path,show_action,None)#get_category_filter(18))

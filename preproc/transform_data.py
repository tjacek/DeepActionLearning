import os
from os import listdir
from os.path import isfile, join

def transform_data(in_path,out_path): 
    all_files = [ f for f in listdir(in_path) if isfile(join(in_path,f)) ]
    for filename in all_files:
        in_file=in_path+filename
        out_file=out_path+filename
        raw_to_tensor(in_file,out_file)
        #print(out_file)

def raw_to_tensor(in_file,out_file):
    out_file=out_file.replace(".bin",".tensor")
    print(out_file)
    cmd="th read-action.lua " + in_file+" "+out_file
    os.system(cmd)

def show_action(action_file):
    cmd="qlua show-action.lua " + action_file
    os.system(cmd)

in_path="/home/user/Desktop/raw_data/"
out_path="/home/user/Desktop/tensor_data/"
scene="a06_s03_e01_sdepth.bin"

#raw_to_tensor(in_path+scene,out_path+scene)
transform_data(in_path,out_path)
